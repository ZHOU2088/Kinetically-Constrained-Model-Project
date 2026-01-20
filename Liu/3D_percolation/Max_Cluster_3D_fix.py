import cupy as cp
import sys
import numpy as np
import cupyx.scipy.ndimage
from scipy.ndimage import generate_binary_structure
import os
from tqdm.notebook import tqdm, trange
import time
from pathlib import Path
import gc
#一次处理一个晶格
#只考虑非周期性边界条件


# 三维情况下的规则1批量应用 CUDA 核函数
apply_rule1_batch_kernel_code_d3_int8 = """
extern "C" __global__
void apply_rule1_batch_kernel(const signed char* initial_lattice, signed char* intermediate_state, 
                             int L, int batch_size) {
    // initial_lattice: 输入的初始晶格状态
    // intermediate_state: 应用规则1后的中间状态
    // Lx, Ly, Lz: 晶格在x, y, z方向的尺寸
    

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int total_z_blocks = (L + blockDim.z - 1) / blockDim.z;
    int batch_idx = blockIdx.z / total_z_blocks;
    int z_block = blockIdx.z % total_z_blocks;
    int z = z_block * blockDim.z + threadIdx.z;

    // 边界检查
    if (batch_idx >= batch_size || x >= L || y >= L || z >= L) return;
    
    int lattice_size = L * L * L;
    int batch_1D_idx = batch_idx * lattice_size;

    int tid = batch_1D_idx + z * L * L + y * L + x;
    
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    // 初始化中间状态
    intermediate_state[tid] = initial_lattice[tid];

    // 规则 1: 如果当前格点是0且所有邻居都是1，则变为1
    if (initial_lattice[tid] == 0) {
        bool all_neighbors_one = true;
        
        for (int i = 0; i < 6; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];
            int neighbor_tid;

            
            if (nx >= 0 && nx < L && ny >= 0 && ny < L && nz >= 0 && nz < L) {
                neighbor_tid = batch_1D_idx + nz * L * L + ny * L + nx;
                if (initial_lattice[neighbor_tid] == 0) {
                    all_neighbors_one = false;
                    break; // 发现一个空位邻居后直接终止搜索
                }
            }
            
        }

        if (all_neighbors_one) {
            intermediate_state[tid] = 1;
        }
    }
}
"""

# 三维情况下的规则2批量应用 CUDA 核函数
apply_rule2_batch_kernel_code_d3_int8 = """
extern "C" __global__
void apply_rule2_batch_kernel(signed char* current_state,
                             int L, int batch_size) {
    
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int total_z_blocks = (L + blockDim.z - 1) / blockDim.z;
    int batch_idx = blockIdx.z / total_z_blocks;
    int z_block = blockIdx.z % total_z_blocks;
    int z = z_block * blockDim.z + threadIdx.z;

    // 边界检查
    if (batch_idx >= batch_size || x >= L || y >= L || z >= L) return;
    
    int lattice_size = L * L * L;
    int batch_1D_idx = batch_idx * lattice_size;

    int tid = batch_1D_idx + z * L * L + y * L + x;
    
    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};
    
    // 规则 2: 如果当前元胞是0，则将其所有邻居设为0
    if (current_state[tid] == 0) {
        for (int i = 0; i < 6; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];
            int neighbor_tid;

            
            if (nx >= 0 && nx < L && ny >= 0 && ny < L && nz >= 0 && nz < L) {
                neighbor_tid = batch_1D_idx + nz * L * L + ny * L + nx;
                current_state[neighbor_tid] = 0;
            }
            
        }
    }
}
"""

# 三维簇大小计数核函数
batch_cluster_count_kernel_code_d3 = """
extern "C" __global__
void count_batch_cluster_sizes_3d(const int* labeled_array, int* size_counts, 
                                 int L, int batch_size, int max_label) {
    int batch_idx = blockIdx.y;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    int lattice_size = L * L * L;
    if (batch_idx >= batch_size || tid >= lattice_size) return;
    
    int global_tid = batch_idx * lattice_size + tid;
    int label = labeled_array[global_tid];
    
    if (label > 0 && label <= max_label) {
        int counts_offset = batch_idx * (max_label + 1);
        atomicAdd(&size_counts[counts_offset + label], 1);
    }
}
"""

# 编译CUDA核函数
rule1_batch_kernel_cupy_d3 = cp.RawKernel(apply_rule1_batch_kernel_code_d3_int8, 'apply_rule1_batch_kernel')
rule2_batch_kernel_cupy_d3 = cp.RawKernel(apply_rule2_batch_kernel_code_d3_int8, 'apply_rule2_batch_kernel')
batch_cluster_count_kernel_cupy_d3 = cp.RawKernel(batch_cluster_count_kernel_code_d3, 'count_batch_cluster_sizes_3d')

class BatchMemoryPool3D:
    """
    三维批量内存池管理类
    
    注意：在三维实现中，batch参数被展宽到blockIdx.z上，
    因此批处理数组的形状为(batch_size, L, L, L)
    """
    
    def __init__(self, L, max_batch_size):
        """
        初始化内存池
        
        参数：
        L: 格点系统尺寸（L×L×L）
        max_batch_size: 最大批处理大小
        max_pool_capacity_multiplier: 池容量乘数，用于创建略大于请求大小的数组
        """
        self.L = L
        self.max_batch_size = max_batch_size
        
        # 初始化三个内存池
        self.batch_lattice_pool = []  # 存储int8类型的三维批处理晶格数组
        self.batch_lattice_label_pool = []  # 存储int32类型的标签数组
        self.batch_int_array_pool = []  # 存储一维整数数组
        
        # 设置每个池的最大存储数量
        self.max_lattice_state_pool_storage_count = 3
        self.max_lattice_label_pool_storage_count = 2
        self.max_int_array_pool_storage_count = 3

    def _get_array_base(self, pool, requested_capacity, array_constructor_func, is_lattice_array=False):
        """
        从内存池获取数组的通用方法
        
        参数：
        pool: 目标内存池
        requested_capacity: 请求的容量
                          - 对于晶格数组：请求的batch_size
                          - 对于一维数组：请求的元素个数
        array_constructor_func: 数组构造函数
        is_lattice_array: 是否为晶格数组（三维或二维）
        
        返回：
        可用的数组
        """
        best_fit_idx = -1
        min_oversize = float('inf')
        
        # 从后往前遍历池，优先检查最近使用的数组
        for i in range(len(pool) - 1, -1, -1):
            arr_base, capacity = pool[i]
            
            # 检查当前数组是否满足容量要求
            if capacity >= requested_capacity:
                oversize = capacity - requested_capacity
                if oversize < min_oversize:
                    min_oversize = oversize
                    best_fit_idx = i
                    if oversize == 0:  # 找到完全匹配的
                        break
        
        # 如果找到合适的数组，从池中取出
        if best_fit_idx != -1:
            arr_base, _ = pool.pop(best_fit_idx)
            return arr_base
        
        # 池中没有合适的数组，创建新的
        # 可以创建略大于请求大小的数组以提高复用率
        return array_constructor_func(requested_capacity)

    def _return_array_base(self, pool, arr_base, max_pool_storage_count):
        """
        将数组返回给内存池
        
        参数：
        pool: 目标内存池
        arr_base: 要返回的数组
        max_pool_storage_count: 池的最大存储数量
        """
        if arr_base is None:
            return
        
        # 获取数组的原始引用（如果传入的是切片）
        original_arr = arr_base.base if arr_base.base is not None else arr_base
        
        if not original_arr.data:  # 检查内存是否已释放
            return
        
        # 如果池未满，将数组加入池中
        if len(pool) < max_pool_storage_count:
            # 计算数组容量
            if original_arr.ndim == 4:  # 三维批处理晶格数组
                capacity = original_arr.shape[0]  # batch_size
            elif original_arr.ndim == 3:  # 二维批处理晶格数组
                capacity = original_arr.shape[0]  # batch_size
            elif original_arr.ndim == 1:  # 一维数组
                capacity = len(original_arr)
            else:
                raise ValueError(f"不支持的数组维度: {original_arr.ndim}")
            
            # 避免重复添加同一数组
            for pooled_arr, _ in pool:
                if pooled_arr is original_arr:
                    return
            pool.append((original_arr, capacity))

    def get_batch_lattice_array(self, batch_size):
        """获取三维批处理晶格数组（int8类型）"""
        def constructor(capacity):
            # 注意：这里使用cp.empty而不是cp.zeros，避免不必要的初始化
            return cp.empty((capacity, self.L, self.L, self.L), dtype=cp.int8)
        
        arr_base = self._get_array_base(
            self.batch_lattice_pool,
            batch_size,
            constructor,
            is_lattice_array=True
        )
        
        # 返回适当大小的切片
        if arr_base.shape[0] > batch_size:
            return arr_base[:batch_size]
        return arr_base

    def return_batch_lattice_array(self, arr):
        """将使用完毕的批处理晶格数组返回到池中"""
        self._return_array_base(
            self.batch_lattice_pool,
            arr,
            self.max_lattice_state_pool_storage_count
        )

    def get_batch_lattice_label_array(self, batch_size):
        """获取批处理标签数组（int32类型）"""
        def constructor(capacity):
            return cp.empty((capacity, self.L, self.L, self.L), dtype=cp.int32)
        
        arr_base = self._get_array_base(
            self.batch_lattice_label_pool,
            batch_size,
            constructor,
            is_lattice_array=True
        )
        
        if arr_base.shape[0] > batch_size:
            return arr_base[:batch_size]
        return arr_base

    def return_batch_lattice_label_array(self, arr):
        """将使用完毕的批处理标签数组返回到池中"""
        self._return_array_base(
            self.batch_lattice_label_pool,
            arr,
            self.max_lattice_label_pool_storage_count
        )

    def get_batch_int_array(self, size):
        """获取一维整数数组（int32类型）"""
        def constructor(capacity):
            return cp.empty(capacity, dtype=cp.int32)
        
        arr_base = self._get_array_base(
            self.batch_int_array_pool,
            size,
            constructor,
            is_lattice_array=False
        )
        
        if len(arr_base) > size:
            return arr_base[:size]
        return arr_base

    def return_batch_int_array(self, arr):
        """将使用完毕的一维整数数组返回到池中"""
        self._return_array_base(
            self.batch_int_array_pool,
            arr,
            self.max_int_array_pool_storage_count
        )

    def clear(self):
        """清空所有内存池"""
        # 清空批处理晶格数组池
        for arr_base, _ in self.batch_lattice_pool:
            del arr_base
        self.batch_lattice_pool.clear()
        
        # 清空批处理标签数组池
        for arr_base, _ in self.batch_lattice_label_pool:
            del arr_base
        self.batch_lattice_label_pool.clear()
        
        # 清空一维数组池
        for arr_base, _ in self.batch_int_array_pool:
            del arr_base
        self.batch_int_array_pool.clear()

    def __del__(self):
        """析构函数，确保内存被正确释放"""
        self.clear()


def run_max_cluster_density_simulation_3d(
    L_list,
    d=3,
    rho_list=None,
    p_list=None,
    num_sample=100,
    periodic=True,
    output_base_dir="E:\JianWen_Zhou\MIS-3D-results\cluster-size-distribution",
    batch_size=5,
    use_p_instead_of_rho=False
):
    """
    三维系统最大独立集模拟函数
    
    参数:
    L_list: 晶格尺寸列表（L是边长，立方晶格L×L×L）
    d: 维度，必须为3
    rho_list: 空置点（0）的概率列表（当use_p_instead_of_rho=False时使用）
    p_list: 占据点（1）的概率列表（当use_p_instead_of_rho=True时使用）
    num_sample: 每个(L, rho/p)组合的样本数量
    periodic: 是否使用周期性边界条件（代码中未实现周期性，保持为非周期性）
    output_base_dir: 输出目录
    batch_size: 批处理大小
    use_p_instead_of_rho: 如果True，使用p_list（占据概率）；如果False，使用rho_list（空置概率）
    """
    if d != 3:
        raise ValueError("此函数专为三维系统设计，d必须为3")
    
    if use_p_instead_of_rho:
        if p_list is None:
            raise ValueError("当use_p_instead_of_rho=True时，必须提供p_list")
        prob_list = p_list
        prob_type = "p"
    else:
        if rho_list is None:
            raise ValueError("当use_p_instead_of_rho=False时，必须提供rho_list")
        prob_list = rho_list
        prob_type = "rho"
    
        

    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_path}")
    print(f"目录是否存在: {output_path.exists()}")
    
    # 定义文件名
    detail_file = output_path / f"max-cluster-density-3d-{prob_type}.txt"
    stat_file = output_path / f"max-cluster-density-stat-3d-{prob_type}.txt"
    
    print(f"详细文件路径: {detail_file}")
    print(f"统计文件路径: {stat_file}")

    # 写入文件头
    try:
        with open(detail_file, 'w') as f:
            f.write(f"L,{prob_type},max_cluster_densities\n")
            f.flush()
        print(f"成功创建详细文件: {detail_file}")
        with open(stat_file, 'w') as f:
            f.write(f"L,{prob_type},mean_density,std_density,second_moment,fourth_moment,Binder_cumulant\n")
            f.flush()
        print(f"成功创建统计文件: {stat_file}")
    except Exception as e:
        print(f"文件创建失败: {e}")
        # 尝试使用当前目录作为备用
        output_path = Path(".")
        detail_file = output_path / f"max-cluster-density-3d-{prob_type}.txt"
        stat_file = output_path / f"max-cluster-density-stat-3d-{prob_type}.txt"
        print(f"使用备用目录: {output_path}")

    device = cp.cuda.Device()
    max_threads_per_block = device.attributes['MaxThreadsPerBlock']
    
    # 生成三维结构元素（6-邻域，冯·诺依曼邻域）
    struct_cpu = generate_binary_structure(rank=3, connectivity=1)
    structure_gpu = cp.array(struct_cpu)
    
    print(f"三维系统批量模拟参数: d={d}, periodic={periodic}, samples={num_sample}")
    print(f"L_list={L_list}, {prob_type}_list={prob_list}")
    print(f"批处理大小: {batch_size}")
    print(f"概率类型: {prob_type} (1=占据概率, 0=空置概率)" if prob_type == "p" else "rho (空置概率)")
    print("-" * 50)
    
    for L in tqdm(L_list, desc="L values"):
        total_sites_N = L**3
        
        # 配置CUDA块和网格
        # 使用三维块结构
        if L >= 8:
            # 使用较小的块大小，例如4x4x4=64线程
            block_s = 4
            block_dim = (block_s, block_s, block_s)
        else:
            block_s = L
            block_dim = (block_s, block_s, block_s)
        
        grid_dim_x = (L + block_dim[0] - 1) // block_dim[0]
        grid_dim_y = (L + block_dim[1] - 1) // block_dim[1]
        z_blocks_per_batch = (L + block_dim[2] - 1) // block_dim[2]
        
        # 注意：您的核函数在z维度上展平了批处理维度
        # blockIdx.z = batch_idx * L + z_block_idx
        # 所以我们需要grid_dim_z_total = L * batch_size
        # 但我们会在每次批次运行时动态计算这个值
        
        batch_memory_pool = BatchMemoryPool3D(L, batch_size)
        
        for prob_val in tqdm(prob_list, desc=f"L={L}", leave=False):
            start_time_prob = time.time()
            max_cluster_densities = []
            
            for batch_start in trange(0, num_sample, batch_size, desc=f"{prob_type}={prob_val:.3f}, L={L}", leave=False):
                batch_end = min(batch_start + batch_size, num_sample)
                current_batch_size = batch_end - batch_start
                
                # 从内存池获取数组
                initial_lattice_batch = batch_memory_pool.get_batch_lattice_array(current_batch_size)
                intermediate_state_batch = batch_memory_pool.get_batch_lattice_array(current_batch_size)
                final_state_batch = batch_memory_pool.get_batch_lattice_array(current_batch_size)
                
                # 初始化晶格
                cp.random.seed()
                random_vals_batch = cp.random.rand(current_batch_size, L, L, L, dtype=cp.float32)
                
                if use_p_instead_of_rho:
                    # p是占据概率：random_val < p -> 1，否则0
                    initial_lattice_batch[:] = (random_vals_batch < prob_val).astype(cp.int8)
                else:
                    # rho是空置概率：random_val > rho -> 1，否则0
                    initial_lattice_batch[:] = (random_vals_batch > prob_val).astype(cp.int8)
                
                # 动态计算网格维度
                # 注意：您的核函数期望blockIdx.z = batch_idx * L + z_block_idx
                # 所以grid_dim_z_total = L * current_batch_size
                grid_dim_z_total = z_blocks_per_batch * current_batch_size
                
                
                # 计算实际的网格维度
                grid_dim_total = (
                    grid_dim_x,  # x方向的块数
                    grid_dim_y,  # y方向的块数
                    grid_dim_z_total  # z方向的块数（包含批次维度）
                )
                
                # 展平数组
                initial_flat = initial_lattice_batch.ravel()
                intermediate_flat = intermediate_state_batch.ravel()
                final_flat = final_state_batch.ravel()
                
                # 规则1：创建中间状态
                rule1_batch_kernel_cupy_d3(
                    grid_dim_total,
                    block_dim,
                    (initial_flat, intermediate_flat, L, current_batch_size)
                )
                
                # 等待规则1完成
                cp.cuda.runtime.deviceSynchronize()
                
                # 规则2：应用中间状态到最终状态
                # 注意：规则2核函数直接修改current_state
                # 所以我们将中间状态复制到最终状态
                cp.copyto(final_state_batch, intermediate_state_batch)
                
                # 应用规则2到最终状态
                rule2_batch_kernel_cupy_d3(
                    grid_dim_total,
                    block_dim,
                    (final_flat, L, current_batch_size)  # 规则2只有一个状态参数
                )
                
                # 等待规则2完成
                cp.cuda.runtime.deviceSynchronize()
                
                # 分析每个样本
                for i in range(current_batch_size):
                    # 获取单个样本
                    sample_final_state = final_state_batch[i]
                    
                    # 标记连通分量
                    labeled_array, num_features = cupyx.scipy.ndimage.label(
                        sample_final_state, structure=structure_gpu)
                    
                    max_cluster_size = 0
                    if num_features > 0:
                        # 计算簇大小
                        max_label_val = int(cp.max(labeled_array))
                        
                        # 从内存池获取计数数组
                        size_counts = batch_memory_pool.get_batch_int_array(max_label_val + 1)
                        # 清零计数数组
                        size_counts[:] = 0
                        
                        # 计算每个簇的大小
                        # 我们需要为每个样本单独调用计数核函数
                        # 首先展平数组
                        labeled_flat = labeled_array.ravel()
                        
                        # 设置网格和块
                        threads_per_block = 256
                        blocks_1d = (total_sites_N + threads_per_block - 1) // threads_per_block
                        
                        # 注意：batch_cluster_count_kernel_cupy_d3 期望批量输入
                        # 我们需要创建适当的网格
                        grid_dim_count = (blocks_1d, 1, 1)  # 对于单样本
                        
                        batch_cluster_count_kernel_cupy_d3(
                            grid_dim_count,
                            (threads_per_block, 1, 1),
                            (labeled_flat, size_counts, L, 1, max_label_val)
                        )
                        
                        cp.cuda.runtime.deviceSynchronize()
                        
                        # 获取最大簇大小
                        # 跳过标签0
                        if max_label_val >= 1:
                            nonzero_mask = size_counts[1:] > 0
                            if cp.any(nonzero_mask):
                                max_cluster_size = int(cp.max(size_counts[1:max_label_val+1]))
                        
                        # 返回计数数组到内存池
                        batch_memory_pool.return_batch_int_array(size_counts)
                    
                    max_cluster_density = max_cluster_size / total_sites_N
                    max_cluster_densities.append(max_cluster_density)
                
                # 返回数组到内存池
                batch_memory_pool.return_batch_lattice_array(initial_lattice_batch)
                batch_memory_pool.return_batch_lattice_array(intermediate_state_batch)
                batch_memory_pool.return_batch_lattice_array(final_state_batch)
                
                # 清理内存
                del random_vals_batch
            
            # 计算统计信息
            if max_cluster_densities:
                max_cluster_densities_np = np.array(max_cluster_densities)
                mean_density = np.mean(max_cluster_densities_np)
                std_density = np.std(max_cluster_densities_np)
                second_moment = np.mean(max_cluster_densities_np**2)
                fourth_moment =  np.mean(max_cluster_densities_np**4)
                Binder_cumulant = 1-(fourth_moment / (3*second_moment**2))
                # 写入详细结果
                try:
                    with open(detail_file, 'a') as f:
                        density_str = ','.join(f"{density:.10e}" for density in max_cluster_densities_np)
                        f.write(f"{L},{prob_val:.7f},{density_str}\n")
                        f.flush()
                    print(f"✓ 详细数据写入成功")
                except Exception as e:
                    print(f"✗ 详细数据写入失败: {e}")    
        
                    # 写入统计信息
                try:
                    with open(stat_file, 'a') as f:
                        f.write(f"{L},{prob_val:.7f},{mean_density:.10e},{std_density:.10e},{second_moment:.10e},{fourth_moment:.10e},{Binder_cumulant:.10e}\n")
                        f.flush()
                    print(f"✓ 统计信息写入成功")
                except Exception as e:
                    print(f"✗ 统计信息写入失败: {e}")
                
                end_time_prob = time.time()
                print(f"L={L}, {prob_type}={prob_val:.4f} 完成，耗时 {end_time_prob - start_time_prob:.2f}s, "
                      f"平均最大密度: {mean_density:.6f}, 样本数: {len(max_cluster_densities)}")
            
            # 清理GPU内存
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
    
    print("三维系统模拟完成!")
    print(f"详细数据保存至: {detail_file}")
    print(f"统计信息保存至: {stat_file}")
    
    # 清理内存池
    if 'batch_memory_pool' in locals():
        batch_memory_pool.clear()

# 辅助函数：创建优化后的概率列表
def create_optimized_probability_list_3d(
    center_prob=0.5,  # 中心概率（相变点附近）
    width=0.2,  # 扫描宽度
    num_points=50,  # 总点数
    concentration=2.0  # 中心点附近的密度
):
    """
    为3D系统创建优化的概率列表，在相变点附近更密集
    
    参数:
    center_prob: 相变点附近的中心概率
    width: 扫描的总宽度
    num_points: 总点数
    concentration: 集中程度因子
    
    返回:
    list: 优化后的概率列表
    """
    prob_min = max(0.0, center_prob - width/2)
    prob_max = min(1.0, center_prob + width/2)
    
    # 创建非线性分布
    t = np.linspace(-1, 1, num_points)
    transformed = np.sign(t) * (np.abs(t) ** concentration)
    probs = center_prob + transformed * (width/2)
    
    # 过滤和舍入
    filtered_probs = sorted(list(set(
        round(x, 6) for x in probs 
        if prob_min <= x <= prob_max
    )))
    
    print(f"生成的概率列表长度: {len(filtered_probs)}")
    print(f"概率范围: {min(filtered_probs):.4f} 到 {max(filtered_probs):.4f}")
    
    return filtered_probs


# 示例使用
if __name__ == "__main__":
    if sys.platform == 'win32':
        cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
        os.environ['PATH'] = f"{cuda_path}\\bin;{os.environ.get('PATH', '')}"
        os.environ['CUDA_PATH'] = cuda_path
    else:
        cuda_path = "/usr/local/cuda-12.4"
        os.environ['PATH'] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
        os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ['CUDA_HOME'] = cuda_path

    # 示例1：使用rho_list（空置概率）
    L_list_3d = [120, 140, 180, 220, 260, 300, 340, 380, 420]  # 较小的L用于测试
    rho_list_3d = create_optimized_probability_list_3d(
        center_prob=0.26,  # 3D系统相变点可能在0.2-0.3附近
        width=0.03,
        num_points=60,
        concentration=3.0
    )
    
    # 运行模拟
    run_max_cluster_density_simulation_3d(
        L_list=L_list_3d,
        d=3,
        rho_list=rho_list_3d,
        num_sample=5000,  # 样本数
        periodic=True,
        output_base_dir="./MIS-3D-results",
        batch_size=1,  # 3D内存占用大，使用较小的batch_size
        use_p_instead_of_rho=False
    )
    