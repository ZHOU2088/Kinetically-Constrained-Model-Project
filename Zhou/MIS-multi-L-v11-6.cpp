// 引入C++标准库，用于输入输出、向量、数值计算、算法、随机数、数学函数等
#include <iostream> // 用于标准输入输出流 (cin, cout, cerr)
#include <vector> // 用于动态数组 (std::vector)
#include <numeric> // 用于数值操作 (如 std::iota, std::accumulate)
#include <algorithm> // 用于通用算法 (如 std::sort, std::unique, std::min, std::max)
#include <random> // 用于生成随机数 (std::mt19937, std::uniform_real_distribution)
#include <cmath> // 用于数学函数 (如 std::sqrt, std::abs, std::pow)
#include <fstream> // 用于文件输入输出流 (std::ifstream, std::ofstream)
#include <string> // 用于字符串操作 (std::string)
#include <sstream> // 用于字符串流 (std::ostringstream)
#include <iomanip> // 用于输入输出控制 (如 std::setprecision, std::fixed)
#include <chrono> // 用于时间测量 (std::chrono)
#include <map> // 用于键值对存储 (std::map,有序)
#include <set> // 用于存储唯一元素 (std::set,有序)
#include <limits> // 用于数值极限 (std::numeric_limits)
#include <unordered_map> // 用于哈希表实现的键值对存储 (std::unordered_map,无序)
#include <omp.h> // 用于OpenMP并行编程


// 预处理指令，检查是否支持 <filesystem> 头文件
#if __has_include(<filesystem>)
#include <filesystem> // 引入文件系统库
namespace fs = std::filesystem; // 定义命名空间别名 fs
#else
// 如果不支持 <filesystem>，则使用 POSIX 和 Windows API 实现目录创建
#include <sys/stat.h> // 用于获取文件状态 (stat)
#ifdef _WIN32
#include <direct.h> // Windows平台下用于目录操作 (_mkdir)
#define MKDIR(path) _mkdir(path) // 定义 MKDIR 宏为 Windows 的 _mkdir
#else
#define MKDIR(path) mkdir(path, 0755) // 定义 MKDIR 宏为 POSIX 的 mkdir，权限为 0755
#endif

// 递归创建目录的函数 (兼容旧版编译器)
void create_directories_recursive(const std::string& path_str) {
    size_t pos = 0; // 当前处理到的路径位置
    std::string current_path; // 当前构建的路径部分
    do {
        pos = path_str.find_first_of("/\\", pos + 1); // 查找下一个路径分隔符 ('/' 或 '\')
        current_path = path_str.substr(0, pos); // 提取到分隔符为止的子路径
        if (!current_path.empty()) { // 如果子路径非空
            #if __has_include(<filesystem>) 
                // 此处的 #if __has_include(<filesystem>) 实际上是冗余的，
                // 因为这个整个 #else 块就是当 <filesystem> 不可用时执行的。
                // 这部分代码块理论上不会被执行，可能是早期代码的残留或笔误。
                // 如果外部的 #if __has_include(<filesystem>) 为 false，
                // 那么内部的这个 #if 也必然为 false。
            #else 
                struct stat st; // 用于存储文件状态信息
                if (stat(current_path.c_str(), &st) == -1) { // 检查路径是否存在，-1 表示不存在或错误
                    MKDIR(current_path.c_str()); // 如果路径不存在，则创建目录
                }
            #endif
        }
    } while (pos != std::string::npos); // 继续直到处理完整个路径字符串
}
#endif


// 定义参数结构体，用于存储模拟所需的各种参数
struct Params {
    int d; // 系统的维度
    int L; // 系统在第一个维度上的长度
    int W; // 系统在其他维度上的长度 (通常 L=W，即超立方体)
    double rho; // 初始晶格中值为0的节点的密度
    bool save_state; // 是否保存每个样本的晶格状态和MIS状态
    unsigned int base_seed; // 用于生成随机数的基准种子
    bool periodic_boundary; // 是否使用周期性边界条件
    std::vector<long long> L_multipliers; // 用于在多维索引和一维索引之间转换的乘数数组
};






// 生成一个双精度浮点数等差数列
// start: 数列起始值
// step: 数列公差
// end: 数列结束值 (包含)
std::vector<double> generateSequence(double start, double step, double end) {
    std::vector<double> result; // 存储生成的数列
    if (step == 0) { // 如果步长为0，无法生成有效序列
        // 返回空向量
        return result;
    }
    if ((step > 0 && start > end) || (step < 0 && start < end)) { // 如果步长方向与起止点矛盾
        // 返回空向量
        return result;
    }
    double current = start; // 当前值初始化为起始值
    if (step > 0) { // 如果步长为正
        while (current <= end + 1e-9) { // 循环直到当前值超过结束值 (加上一个小的容差以处理浮点精度问题)
            result.push_back(current); // 将当前值加入结果向量
            current += step; // 更新当前值
        }
    } else { // 如果步长为负
        while (current >= end - 1e-9) { // 循环直到当前值小于结束值 (加上一个小的容差)
            result.push_back(current); // 将当前值加入结果向量
            current += step; // 更新当前值 (注意 step 是负数)
        }
    }
    // 注意：对于浮点数，直接比较可能会因为精度问题导致结果不符合预期。
    // 1e-9 是一个常用的epsilon值，用于容忍这种微小的计算误差。
    // 例如，如果 end 是 1.0，step 是 0.1，current 可能会变成 0.9999999999999999，
    // 此时 current <= end 仍然成立。
    // 或者 current 变成 1.0000000000000001，此时 current <= end + 1e-9 仍然可能成立，
    // 确保 end 本身被包含。
    return result; // 返回生成的数列
}


// 生成一个整数等差数列
// start: 数列起始值
// step: 数列公差
// end: 数列结束值 (包含)
std::vector<int> generateSequence_int(int start, int step, int end) {
    std::vector<int> result; // 存储生成的数列
    if (step == 0) { return result; } // 如果步长为0，返回空向量
    if ((step > 0 && start > end) || (step < 0 && start < end)) { return result; } // 如果步长方向与起止点矛盾，返回空向量
    int current = start; // 当前值初始化为起始值
    if (step > 0) { // 如果步长为正
        while (current <= end) { // 循环直到当前值超过结束值
            result.push_back(current); // 将当前值加入结果向量
            current += step; // 更新当前值
        }
    } else { // 如果步长为负
        while (current >= end) { // 循环直到当前值小于结束值
            result.push_back(current); // 将当前值加入结果向量
            current += step; // 更新当前值
        }
    }
    // 对于整数，不需要容差。
    return result; // 返回生成的数列
}


// 将一维索引转换为 d 维坐标
// index: 一维索引值
// L_param: 第0维的长度
// W_param: 其他维度的长度
// d_param: 维度数
std::vector<int> index_to_coords(int index, int L_param, int W_param, int d_param) {
    std::vector<int> coords(d_param); // 初始化坐标向量，大小为维度数
    if (d_param == 0) return coords; // 0维系统，返回空坐标
    if (d_param == 1) { // 1维系统
        if (L_param <= 0) return coords; // 长度无效，返回空坐标（或未定义行为）
        coords[0] = index % L_param; // 坐标即为索引模长度 (处理周期性)
        return coords;
    }

    long long current_index = index; // 使用 long long 避免中间计算溢出
    
    // 计算最后一个维度的坐标 (d_param - 1)
    coords[d_param - 1] = current_index % W_param;
    current_index /= W_param;

    // 计算中间维度 (从 d_param - 2 到 1) 的坐标
    for (int i = d_param - 2; i >= 1; --i) {
        coords[i] = current_index % W_param;
        current_index /= W_param;
    }
    
    // 计算第一个维度 (0) 的坐标
    if (d_param > 0) { // 确保维度至少为1
         coords[0] = current_index % L_param;
    }
    return coords; // 返回计算得到的坐标向量
}


// 将 d 维坐标转换为一维索引
// coords: d 维坐标向量
// L_param: 第0维的长度
// W_param: 其他维度的长度
// d_param: 维度数
int coords_to_index(const std::vector<int>& coords, int L_param, int W_param, int d_param) {
    if (d_param == 0) return 0; // 0维系统，索引为0
    if (d_param == 1) { // 1维系统
        if (L_param <= 0) return 0; // 长度无效，返回0（或未定义行为）
        return coords[0] % L_param; // 索引即为坐标模长度 (处理周期性)
    }

    // 索引计算公式: index = (...((c0*W + c1)*W + c2)*W + ...)*W + c(d-1)
    // 其中 c0 的范围是 L_param，其他 ci 的范围是 W_param
    int index = coords[0]; // 从第0维坐标开始
    for (int i = 1; i < d_param; ++i) { // 遍历其余维度
        index = index * W_param + coords[i]; // 累积计算索引值
    }
    return index; // 返回计算得到的一维索引
}


// // (这是一个被注释掉的旧版本函数)
// // 优化获取邻居节点的函数 (旧版本)
// // index: 当前节点的一维索引
// // p: 参数结构体
// // neighbors_output: 用于存储邻居索引的向量 (会被清空和填充)
// void get_neighbors_optimized(int index, const Params& p, std::vector<int>& neighbors_output) {
//     neighbors_output.clear(); // 清空输出向量

//     // 基本的有效性检查
//     if (p.d == 0 || p.L <= 0 || (p.d > 1 && p.W <= 0)) return; // 如果维度为0，或L/W无效，则无邻居
    
//     // 标记是否需要对邻居列表进行排序和去重
//     // (主要在周期性边界且维度尺寸为2时，正向和反向邻居可能是同一个)
//     bool needs_sort_unique = false; 

//     // 遍历每个维度
//     for (int i = 0; i < p.d; ++i) {
//         long long multiplier = p.L_multipliers[i]; // 当前维度在索引计算中的乘数/权重
//         int dim_size = (i == 0) ? p.L : p.W; // 当前维度的尺寸

//         if (dim_size <= 1) continue; // 如果维度尺寸为1或更小，该维度上没有不同邻居

//         // 计算当前节点在当前维度 i 上的坐标值
//         int current_coord_val;
//         long long temp_index = index; // 临时索引变量
//         if (p.d == 1) { // 1维情况
//             current_coord_val = temp_index % p.L;
//         } else { // 多维情况
//             // (旧的坐标计算逻辑，较为复杂)
//             // 如果是第0维
//             if (i == 0) { 
//                 long long div_factor = 1; // 除数因子
//                 for(int k=1; k<p.d; ++k) div_factor *= p.W; // 计算 W^(d-1)
//                 current_coord_val = (temp_index / div_factor) % p.L; // (index / W^(d-1)) % L
//             } else { // 如果是其他维度 (i > 0)
//                 long long div_factor = 1; // 除数因子
//                 for(int k=i+1; k<p.d; ++k) div_factor *= p.W; // 计算 W^(d-1-i)
//                 current_coord_val = (temp_index / div_factor) % p.W; // (index / W^(d-1-i)) % W
//             }
//         }

//         // 如果是周期性边界且维度尺寸为2，可能需要排序去重
//         if (p.periodic_boundary && dim_size == 2) {
//             needs_sort_unique = true; 
//         }

//         // 处理正方向的邻居
//         if (p.periodic_boundary) { // 周期性边界
//             if (current_coord_val == dim_size - 1) { // 如果在正向边界
//                 neighbors_output.push_back(index - current_coord_val * multiplier); // 邻居是该维度坐标为0的节点
//             } else { // 不在边界
//                 neighbors_output.push_back(index + multiplier); // 邻居是坐标+1的节点
//             }
//         } else { // 非周期性边界 (硬墙)
//             if (current_coord_val + 1 < dim_size) { // 如果正方向上还有节点
//                 neighbors_output.push_back(index + multiplier); // 添加正向邻居
//             }
//         }

//         // 处理负方向的邻居
//         if (p.periodic_boundary) { // 周期性边界
//             if (current_coord_val == 0) { // 如果在负向边界
//                 neighbors_output.push_back(index + (dim_size - 1) * multiplier); // 邻居是该维度坐标为 dim_size-1 的节点
//             } else { // 不在边界
//                 neighbors_output.push_back(index - multiplier); // 邻居是坐标-1的节点
//             }
//         } else { // 非周期性边界 (硬墙)
//             if (current_coord_val > 0) { // 如果负方向上还有节点
//                 neighbors_output.push_back(index - multiplier); // 添加负向邻居
//             }
//         }
//     }

//     // 如果需要，对邻居列表进行排序和去重
//     if (needs_sort_unique) {
//          std::sort(neighbors_output.begin(), neighbors_output.end()); // 排序
//          neighbors_output.erase(std::unique(neighbors_output.begin(), neighbors_output.end()), neighbors_output.end()); // 去重
//     }
// }


// 优化获取邻居节点的函数 (当前使用版本)
// index: 当前节点的一维索引
// p: 参数结构体
// neighbors_output: 用于存储邻居索引的向量 (会被清空和填充)
void get_neighbors_optimized(int index, const Params& p, std::vector<int>& neighbors_output) {
    neighbors_output.clear(); // 清空输出向量

    // 基本的有效性检查
    if (p.d == 0 || p.L <= 0 || (p.d > 1 && p.W <= 0)) return; // 如果维度为0，或L/W无效，则无邻居

    // 标记是否需要对邻居列表进行排序和去重


    for (int i = 0; i < p.d; ++i) { // 遍历每个维度
        long long step_multiplier = p.L_multipliers[i]; // 当前维度在索引计算中的“步长”或“权重”
                                                       // 这个值等于 W^(d-1-i) (对于维度i>0) 或 L等效值(对于维度0,如果L!=W)
                                                       // 这也是获取当前维度坐标值时所需的除数因子（用于隔离该坐标的“块”）
        int dim_size = (i == 0) ? p.L : p.W; // 当前维度的尺寸

        if (dim_size <= 1) continue; // 如果维度尺寸为1或更小，该维度上没有不同邻居

        int current_coord_val; // 当前节点在当前维度 i 上的坐标值
        // 计算当前维度 'i' 的坐标值
        // 用于分离出维度 'i' 坐标块的除数就是 p.L_multipliers[i] 本身。
        // 例子: index = c0*P0 + c1*P1 + c2*P2. 其中 P0=W^2, P1=W, P2=1 (假设L=W).
        // c0 = (index / P0) % L_size (L_size 是第0维的大小)
        // c1 = (index / P1) % W_size (W_size 是第1维的大小)
        // c2 = (index / P2) % W_size (W_size 是第2维的大小)
        // 这正是旧版本中 div_factor 循环所计算的内容。
        if (p.d == 1) { // 1维特殊情况, p.L_multipliers[0] 通常是 1.
            current_coord_val = index % p.L;
        } else {
            // 对于任何维度 i, p.L_multipliers[i] 是正确的除数
            // 用 index 除以它得到该坐标对应的“块索引”，然后模当前维度尺寸得到坐标值。
            current_coord_val = (index / p.L_multipliers[i]) % dim_size;
        }



        // 正方向的邻居
        if (p.periodic_boundary) { // 周期性边界条件
            if (current_coord_val == dim_size - 1) { // 如果在正向边界
                // 邻居是该维度坐标为0的节点 (index - coord_val * multiplier 会将该维度的贡献清零)
                neighbors_output.push_back(index - current_coord_val * step_multiplier); 
            } else { // 不在边界
                neighbors_output.push_back(index + step_multiplier); // 邻居是坐标+1的节点
            }
        } else { // 非周期性边界 (硬墙)
            if (current_coord_val + 1 < dim_size) { // 如果正方向上还有节点
                neighbors_output.push_back(index + step_multiplier); // 添加正向邻居
            }
        }

        // 负方向的邻居
        if (p.periodic_boundary) { // 周期性边界条件
            if (current_coord_val == 0) { // 如果在负向边界
                // 邻居是该维度坐标为 (dim_size-1) 的节点
                neighbors_output.push_back(index + (dim_size - 1) * step_multiplier); 
            } else { // 不在边界
                neighbors_output.push_back(index - step_multiplier); // 邻居是坐标-1的节点
            }
        } else { // 非周期性边界 (硬墙)
            if (current_coord_val > 0) { // 如果负方向上还有节点
                neighbors_output.push_back(index - step_multiplier); // 添加负向邻居
            }
        }
    }

}











// 生成初始晶格状态
// total_nodes: 晶格中的总节点数
// rho: 节点值为0的初始密度 (值为1的密度是 1-rho)
// gen: 随机数生成器
std::vector<int> generate_initial_lattice(long long total_nodes, double rho, std::mt19937& gen) {
    std::vector<int> lattice(total_nodes); // 创建晶格向量
    std::uniform_real_distribution<> distrib(0.0, 1.0); // 创建一个0到1之间的均匀实数分布
    for (long long i = 0; i < total_nodes; ++i) { // 遍历每个节点
        // 如果生成的随机数小于 rho，则节点值为0，否则为1
        lattice[i] = (distrib(gen) < rho) ? 0 : 1; 
    }
    return lattice; // 返回生成的初始晶格
}


// 应用模型规则来更新晶格状态
// lattice: 当前晶格状态 (会被修改)
// p: 参数结构体
// total_nodes: 晶格总节点数
// neighbor_cache: 用于临时存储邻居列表的向量，避免重复分配内存
void apply_rules(std::vector<int>& lattice, const Params& p, long long total_nodes, std::vector<int>& neighbor_cache) { 
    if (total_nodes == 0) return; // 如果没有节点，则不执行任何操作

    // 规则1: 如果一个节点是0，并且其所有邻居都是1，则该节点变成1
    std::vector<long long> nodes_to_become_one; // 存储将要从0变为1的节点索引
    if (p.rho > 0.0 && total_nodes > 0) { // 预估容量以提高效率，仅当rho>0 (即可能有0存在时)
        nodes_to_become_one.reserve(static_cast<size_t>(total_nodes * (1.0 - p.rho) * 0.1 + 10)); // 粗略估计
    }

    for (long long i = 0; i < total_nodes; ++i) { // 遍历所有节点
        if (lattice[i] == 0) { // 如果当前节点是0
            get_neighbors_optimized(i, p, neighbor_cache); // 获取其邻居

            // 特殊情况处理：如果一个节点没有邻居 (例如孤立节点或1x1x...x1系统)
            if (neighbor_cache.empty() && total_nodes > 1) { // 多于一个节点但没有邻居，则此0节点不满足变1条件
                continue;
            }

            bool all_neighbors_one = true; // 假设所有邻居都是1
            if (neighbor_cache.empty() && total_nodes == 1) { // 单节点系统，0节点没有邻居，不满足“所有邻居为1”
                all_neighbors_one = false; 
            }

            for (int neighbor_idx : neighbor_cache) { // 检查所有邻居
                if (lattice[neighbor_idx] == 0) { // 如果有任何一个邻居是0
                    all_neighbors_one = false; // 则不满足条件
                    break; // 无需再检查其他邻居
                }
            }

            if (all_neighbors_one && !neighbor_cache.empty()) { // 如果所有邻居都是1且至少有一个邻居
                nodes_to_become_one.push_back(i); // 将此节点加入待变1列表
            }
        }
    }

    // 执行状态更新：将满足条件的0节点变为1
    for (long long node_idx : nodes_to_become_one) {
        lattice[node_idx] = 1;
    }

    // 规则2: 如果一个节点是0，则其所有邻居都变成0 (侵蚀规则)
    // 注意：这个规则是在规则1之后应用的，并且是基于规则1更新后的晶格状态
    std::vector<char> to_be_zeroed(total_nodes, 0); // 标记将要变为0的节点 (使用char作为布尔标记节省空间)
    for (long long i = 0; i < total_nodes; ++i) { // 再次遍历所有节点
        if (lattice[i] == 0) { // 如果当前节点是0 (可能是初始就是0，或在规则1后仍为0)
            get_neighbors_optimized(i, p, neighbor_cache); // 获取其邻居
            for (int neighbor_idx : neighbor_cache) { // 遍历所有邻居
                if (neighbor_idx >= 0 && neighbor_idx < total_nodes) { // 确保邻居索引有效
                    to_be_zeroed[neighbor_idx] = 1; // 标记该邻居将要变为0
                }
            }
        }
    }

    // 执行状态更新：将被标记的节点变为0
    for(long long i = 0; i < total_nodes; ++i) {
        if(to_be_zeroed[i]) { // 如果节点被标记
            lattice[i] = 0; // 将其值设为0
        }
    }
}


// Hopcroft-Karp算法命名空间，用于寻找二分图的最大基数匹配 (Maximum Cardinality Matching, MCM)
namespace HopcroftKarp { 
    const int INF = std::numeric_limits<int>::max(); // 定义无穷大，用于距离计算

    // Hopcroft-Karp算法结果结构体
    struct HKResult {
        int matching_size; // 找到的最大匹配的大小 (即匹配边的数量)
        std::vector<int> match_U; // U集合中每个节点的匹配对象在V中的索引 (-1表示未匹配)
        std::vector<int> match_V; // V集合中每个节点的匹配对象在U中的索引 (-1表示未匹配)
    };

    // BFS阶段：构建增广路径的层次图
    // current_adj_U_to_V: U到V的邻接表
    // match_U: U中节点的当前匹配情况
    // match_V: V中节点的当前匹配情况
    // dist: U中节点到未匹配V节点的最短增广路径长度 (层次)
    // nU_hk: U集合的节点数
    // nV_hk: V集合的节点数
    // 返回值: 是否找到了到未匹配V节点的增广路径
    bool bfs(const std::vector<std::vector<int>>& current_adj_U_to_V,
             const std::vector<int>& match_U,
             std::vector<int>& match_V,
             std::vector<int>& dist,
             int nU_hk, int nV_hk) {
        std::fill(dist.begin(), dist.end(), INF); // 初始化所有U中节点的距离为无穷大
        std::vector<int> q; // BFS队列
        q.reserve(nU_hk); // 预分配队列空间

        // 将所有U中未匹配的节点加入队列，距离设为0
        for (int u = 0; u < nU_hk; ++u) {
            if (match_U[u] == -1) { // 如果u未匹配
                q.push_back(u);
                dist[u] = 0;
            }
        }

        int head = 0; // 队列头部指针
        bool found_augmenting_path_to_unmatched_V = false; // 标记是否找到通往未匹配V节点的路径
        while(head < q.size()){ // 当队列非空
            int u = q[head++]; // 取出队首节点u
            for (int v_node_in_V : current_adj_U_to_V[u]) { // 遍历u在V中的邻居v
                if (v_node_in_V < 0 || v_node_in_V >= nV_hk) continue; // 邻居索引无效则跳过

                if (match_V[v_node_in_V] == -1) { // 如果v未匹配
                    found_augmenting_path_to_unmatched_V = true; // 找到了到未匹配V节点的路径
                }
                // 如果v未匹配，或者v已匹配的节点u' (match_V[v_node_in_V]) 尚未被访问 (dist[u'] == INF)
                // 这意味着可以通过 (u,v) (非匹配边) 和 (v, u') (匹配边) 扩展增广路径
                if (match_V[v_node_in_V] == -1 ||
                    (match_V[v_node_in_V] < nU_hk && dist[match_V[v_node_in_V]] == INF) ) {
                     if(match_V[v_node_in_V] != -1) { // 如果v已匹配 (即 match_V[v_node_in_V] != -1)
                        dist[match_V[v_node_in_V]] = dist[u] + 1; // 更新u'的距离
                        q.push_back(match_V[v_node_in_V]); // 将u'加入队列
                     }
                }
            }
        }
        return found_augmenting_path_to_unmatched_V; // 返回是否找到任何到未匹配V节点的路径
    }

    // DFS阶段：在BFS构建的层次图上寻找不相交的增广路径
    // u: 当前DFS搜索的U中节点
    // (其他参数同bfs)
    // 返回值: 是否为节点u找到了一条增广路径
    bool dfs(int u, const std::vector<std::vector<int>>& current_adj_U_to_V,
             std::vector<int>& match_U, std::vector<int>& match_V, std::vector<int>& dist,
             int nU_hk, int nV_hk) {
        for (int v_node_in_V : current_adj_U_to_V[u]) { // 遍历u在V中的邻居v
            if (v_node_in_V < 0 || v_node_in_V >= nV_hk) continue; // 邻居索引无效则跳过

            // 如果v未匹配，或者v已匹配的节点u' (match_V[v_node_in_V]) 满足层次条件 (dist[u'] == dist[u] + 1)
            // 并且从u'出发能找到增广路径 (递归调用dfs)
            if (match_V[v_node_in_V] == -1 ||
                (match_V[v_node_in_V] < nU_hk && dist[match_V[v_node_in_V]] == dist[u] + 1 &&
                 dfs(match_V[v_node_in_V], current_adj_U_to_V, match_U, match_V, dist, nU_hk, nV_hk))) {
                match_V[v_node_in_V] = u; // 更新匹配：v匹配到u
                match_U[u] = v_node_in_V; // 更新匹配：u匹配到v
                return true; // 找到一条增广路径
            }
        }
        dist[u] = INF; // 如果从u出发未找到增广路径，将其距离设为INF，表示不再从该路径访问
        return false; // 未找到增广路径
    }

    // 执行Hopcroft-Karp算法主函数
    // num_U: U集合的节点数
    // num_V: V集合的节点数
    // adj_list_U: U到V的邻接表 (adj_list_U[u] 是 u 的邻居列表)
    // 返回值: HKResult结构体，包含最大匹配大小和匹配详情
    HKResult run(int num_U, int num_V, const std::vector<std::vector<int>>& adj_list_U) {
        std::vector<int> current_match_U(num_U, -1); // U中节点的匹配，初始化为-1 (未匹配)
        std::vector<int> current_match_V(num_V, -1); // V中节点的匹配，初始化为-1 (未匹配)
        std::vector<int> current_dist; // U中节点的距离数组
        if (num_U > 0) current_dist.resize(num_U); // 如果U非空，调整大小

        int matching_size = 0; // 初始化最大匹配大小
        // 主循环：只要BFS还能找到到未匹配V节点的增广路径
        while (num_U > 0 && num_V > 0 && bfs(adj_list_U, current_match_U, current_match_V, current_dist, num_U, num_V)) {
            for (int u = 0; u < num_U; ++u) { // 遍历所有U中节点
                // 如果u未匹配，并且能从u通过DFS找到一条增广路径
                if (current_match_U[u] == -1 && dfs(u, adj_list_U, current_match_U, current_match_V, current_dist, num_U, num_V)) {
                    matching_size++; // 匹配大小加1
                }
            }
        }
        return {matching_size, current_match_U, current_match_V}; // 返回结果
    }
}


// 根据最大匹配结果，使用Konig定理相关方法计算最大独立集(MIS)的节点索引
// current_nU: 当前二分图U部分的节点数 (局部索引)
// current_nV: 当前二分图V部分的节点数 (局部索引)
// adj_U_to_V_graph: U到V的邻接表 (使用局部索引)
// current_match_U: U中节点的匹配情况 (局部索引)
// current_match_V: V中节点的匹配情况 (局部索引)
// U_global_indices: U中局部索引到全局节点索引的映射
// V_global_indices: V中局部索引到全局节点索引的映射
// 返回值: 包含在最大独立集中的节点的全局索引列表
std::vector<int> get_mis_nodes_indices( 
    int current_nU, int current_nV,
    const std::vector<std::vector<int>>& adj_U_to_V_graph,
    const std::vector<int>& current_match_U,
    const std::vector<int>& current_match_V,
    const std::vector<int>& U_global_indices,
    const std::vector<int>& V_global_indices
) {
    // Z_U_local_bool: 标记U中的节点是否属于集合Z (从U中未匹配点出发，沿交替路径可达的U中节点)
    std::vector<char> Z_U_local_bool(current_nU, 0); 
    // Z_V_local_bool: 标记V中的节点是否属于集合Z (从U中未匹配点出发，沿交替路径可达的V中节点)
    std::vector<char> Z_V_local_bool(current_nV, 0);
    std::vector<int> q_bfs; // BFS队列，用于寻找集合Z
    if (current_nU + current_nV > 0) { // 预分配队列空间
      q_bfs.reserve(current_nU + current_nV);
    }

    // visited_U_for_Z/visited_V_for_Z: BFS过程中标记节点是否已访问，避免重复处理
    std::vector<char> visited_U_for_Z(current_nU, 0);
    std::vector<char> visited_V_for_Z(current_nV, 0);

    // 从U中所有未匹配的节点开始BFS，构建集合Z
    for (int u_local = 0; u_local < current_nU; ++u_local) {
        if (current_match_U[u_local] == -1 && !visited_U_for_Z[u_local]) { // 如果u_local未匹配且未访问
            q_bfs.clear(); // 清空队列开始新的BFS
            q_bfs.push_back(u_local); // 将u_local加入队列
            visited_U_for_Z[u_local] = 1; // 标记已访问
            Z_U_local_bool[u_local] = 1; // u_local属于Z_U

            int head = 0; // 队列头指针
            while(head < q_bfs.size()){ // 当队列非空
                int curr_u_local = q_bfs[head++]; // 取出队首U节点

                // 检查curr_u_local的有效性，以防adj_U_to_V_graph大小不一致 (防御性编程)
                if (curr_u_local < 0 || curr_u_local >= static_cast<int>(adj_U_to_V_graph.size())) continue;

                // 遍历curr_u_local的邻居v (这些是图中的边，不一定是匹配边)
                // 路径: U_unmatched --(non-match)--> V --(match)--> U --(non-match)--> V ...
                // 从 Z_U 中的 curr_u_local 出发，通过非匹配边到达 V 中的 neighbor_v_local
                for (int neighbor_v_local : adj_U_to_V_graph[curr_u_local]) {
                    if (neighbor_v_local < 0 || neighbor_v_local >= current_nV) continue; // V节点索引无效
                    if (!visited_V_for_Z[neighbor_v_local]) { // 如果v未访问
                        visited_V_for_Z[neighbor_v_local] = 1; // 标记v已访问
                        Z_V_local_bool[neighbor_v_local] = 1; // v属于Z_V

                        // 如果v已匹配到某个u' (next_u_local)，则通过匹配边从v到达u'
                        if (current_match_V[neighbor_v_local] != -1 && // v已匹配
                            current_match_V[neighbor_v_local] < current_nU && // 匹配的U节点索引有效
                            !visited_U_for_Z[current_match_V[neighbor_v_local]]) { // 且u'未访问
                             int next_u_local = current_match_V[neighbor_v_local]; // 获取u'
                             visited_U_for_Z[next_u_local] = 1; // 标记u'已访问
                             Z_U_local_bool[next_u_local] = 1; // u'属于Z_U
                             q_bfs.push_back(next_u_local); // 将u'加入队列继续BFS
                        }
                    }
                }
            }
        }
    }

    std::vector<int> mis_global_indices; // 存储最大独立集中节点的全局索引
    mis_global_indices.reserve(current_nU + current_nV); // 预估容量

    // 根据Konig定理的推论：MIS = (U ∩ Z) ∪ (V \ Z)
    // 或者等价地，MIS = Z_U ∪ (V \ Z_V)
    // 其中 Z_U 是指 Z ∩ U，Z_V 是指 Z ∩ V。
    // 最小顶点覆盖 VC = (U \ Z_U) ∪ Z_V
    // 最大独立集 MIS 是 VC 的补集。
    // MIS = (U U V) \ VC = (U U V) \ ( (U \ Z_U) U Z_V )
    //     = (U \ (U \ Z_U)) U (V \ Z_V)  (假设U,V不交)
    //     = Z_U U (V \ Z_V)

    // 添加 Z_U 中的节点 (即 Z_U_local_bool[u_local] 为 true 的节点)
    for (int u_local = 0; u_local < current_nU; ++u_local) {
        if (Z_U_local_bool[u_local]) { 
            mis_global_indices.push_back(U_global_indices[u_local]); // 添加其全局索引
        }
    }
    // 添加 V \ Z_V 中的节点 (即 Z_V_local_bool[v_local] 为 false 的节点)
    for (int v_local = 0; v_local < current_nV; ++v_local) {
        if (!Z_V_local_bool[v_local]) { 
            mis_global_indices.push_back(V_global_indices[v_local]); // 添加其全局索引
        }
    }

    std::sort(mis_global_indices.begin(), mis_global_indices.end()); // 对MIS节点索引排序
    return mis_global_indices; // 返回MIS节点列表
}


// 格式化rho值为字符串，去除末尾多余的0和小数点
// rho_val: 要格式化的rho值 (double类型)
std::string format_rho_string(double rho_val) { 
    std::ostringstream oss; // 创建字符串输出流
    oss << std::fixed << std::setprecision(10) << rho_val; // 设置固定点表示法和10位小数精度
    std::string s = oss.str(); // 转换为字符串
    s.erase(s.find_last_not_of('0') + 1, std::string::npos); // 删除末尾所有的'0'
    if (!s.empty() && s.back() == '.') { // 如果删除'0'后末尾是'.'
        s.pop_back(); // 删除末尾的'.'
    }
    return s; // 返回格式化后的字符串
}


// 将晶格状态和MIS状态保存到文件
// lattice_array: 晶格状态数组 (0或1)
// mis_array: MIS状态数组 (0或1, 表示节点是否在MIS中)
// p: 参数结构体
// sample_seed_for_filename: 用于文件名的样本种子号
void save_state_to_file(const std::vector<int>& lattice_array, 
                       const std::vector<int>& mis_array,
                       const Params& p, unsigned int sample_seed_for_filename) {
    std::string rho_str = format_rho_string(p.rho); // 格式化rho值
    std::string dir = "MIS-graph/mis-data"; // 定义输出目录
    // 构建文件名
    std::string filename = dir + "/combined-state-d=" + std::to_string(p.d) +
                           "-L=" + std::to_string(p.L) + "-W=" + std::to_string(p.W) +
                           "-rho=" + rho_str + "-seed=" + std::to_string(sample_seed_for_filename) + ".txt";

    // 检查并创建目录
    #if __has_include(<filesystem>) // 如果支持<filesystem>
        if (!fs::exists(dir)) { // 如果目录不存在
            fs::create_directories(dir); // 创建目录 (包括所有父目录)
        }
    #else // 如果不支持<filesystem>
        create_directories_recursive(dir); // 使用自定义的递归创建目录函数
    #endif

    std::ofstream outfile(filename); // 打开输出文件流
    if (!outfile.is_open()) { // 如果文件打开失败
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl; // 输出错误信息
        return; // 提前返回
    }
    // 写入晶格状态
    outfile << "Lattice:" << std::endl;
    if (!lattice_array.empty()) {
        outfile << lattice_array[0]; // 写入第一个元素
        for (size_t i = 1; i < lattice_array.size(); ++i) {
            outfile << " " << lattice_array[i]; // 写入后续元素，用空格分隔
        }
    }
    outfile << std::endl;
    // 写入MIS状态
    outfile << "MIS:" << std::endl;
    if (!mis_array.empty()) {
        outfile << mis_array[0]; // 写入第一个元素
        for (size_t i = 1; i < mis_array.size(); ++i) {
            outfile << " " << mis_array[i]; // 写入后续元素，用空格分隔
        }
    }
    outfile << std::endl;
    outfile.close(); // 关闭文件
}


// 存储MIS计算结果的结构体
struct MisResult {
    std::vector<int> mis_nodes_global_indices; // MIS中节点的全局索引列表
    std::vector<int> mis_lattice_representation; // MIS的晶格表示 (1表示在MIS中, 0表示不在)
    int nU; // 二分图U部分的节点数
    int nV; // 二分图V部分的节点数
    int mcm_size; // 最大匹配的大小
    std::vector<int> U_local_to_global_idx; // U部分局部索引到全局索引的映射
    std::vector<int> V_local_to_global_idx; // V部分局部索引到全局索引的映射
    std::vector<std::vector<int>> final_adj_U_to_V_local; // U到V的邻接表 (使用局部索引)
};


// 计算给定晶格的MIS表示
// lattice: 输入的晶格状态 (0或1)
// p: 参数结构体
// total_nodes: 晶格总节点数
// neighbor_cache: 邻居计算的缓存
// 返回值: MisResult结构体，包含MIS的详细信息
MisResult calculate_mis_representation(
    const std::vector<int>& lattice,
    const Params& p,
    long long total_nodes,
    std::vector<int>& neighbor_cache 
) {
    MisResult result; // 初始化结果结构体
    result.mis_lattice_representation.assign(total_nodes, 0); // 初始化MIS晶格表示全为0

    // 映射：全局索引 -> U/V局部索引 (-1表示不属于该部分或不是活动节点)
    std::vector<int> global_to_U_local_map(total_nodes, -1);
    std::vector<int> global_to_V_local_map(total_nodes, -1);

    // 预估活动节点数量 (值为1的节点)，用于预分配向量内存
    size_t active_nodes_estimate = (total_nodes > 0) ? static_cast<size_t>(total_nodes * (1.0 - p.rho) * 1.1 + 10) : 10; 
    active_nodes_estimate = std::max(active_nodes_estimate, static_cast<size_t>(10)); // 保证最小容量
    result.U_local_to_global_idx.reserve(active_nodes_estimate / 2 + 1); 
    result.V_local_to_global_idx.reserve(active_nodes_estimate / 2 + 1);

    // 遍历所有节点，将值为1的节点（活动节点）根据其坐标和的奇偶性划分到U或V集合
    for (long long i = 0; i < total_nodes; ++i) { // i 是全局索引
        if (lattice[i] == 1) { // 如果是活动节点
            // 计算节点i的各维度坐标之和
            int coord_sum_optimized = 0;
            long long temp_idx_for_sum = i; // 临时索引变量
            if (p.d == 0) { // 0维情况
                 // 坐标和为0 (或未定义，但通常单个节点视为偶数)
            } else if (p.d == 1) { // 1维情况
                coord_sum_optimized = temp_idx_for_sum % p.L; // 坐标即索引
            } else { // 多维情况
                // 从最低维(d-1)开始加到坐标和
                coord_sum_optimized += temp_idx_for_sum % p.W; // coord[d-1]
                temp_idx_for_sum /= p.W;
                
                // 中间维度 (d-2 到 1)
                for (int k_dim = p.d - 2; k_dim >= 1; --k_dim) {
                    coord_sum_optimized += temp_idx_for_sum % p.W; // coord[k_dim]
                    temp_idx_for_sum /= p.W;
                }
                
                // 最高维(0)
                coord_sum_optimized += temp_idx_for_sum % p.L; // coord[0]
            }

            // 根据坐标和的奇偶性分配到U或V
            if (coord_sum_optimized % 2 == 0) { // 偶数和 -> U集合
                global_to_U_local_map[i] = result.U_local_to_global_idx.size(); // 记录全局到局部U的映射
                result.U_local_to_global_idx.push_back(i); // 添加到U的全局索引列表
            } else { // 奇数和 -> V集合
                global_to_V_local_map[i] = result.V_local_to_global_idx.size(); // 记录全局到局部V的映射
                result.V_local_to_global_idx.push_back(i); // 添加到V的全局索引列表
            }
        }
    }

    result.nU = result.U_local_to_global_idx.size(); // U集合大小
    result.nV = result.V_local_to_global_idx.size(); // V集合大小
    if (result.nU > 0) { // 如果U非空，初始化U到V的邻接表
        result.final_adj_U_to_V_local.assign(result.nU, std::vector<int>());
    }

    // 构建二分图的邻接表 (U中节点到V中节点的边)
    if (result.nU > 0 && result.nV > 0) { // 只有当U和V都非空时才可能有边
        for (int u_local_idx = 0; u_local_idx < result.nU; ++u_local_idx) { // 遍历U中的每个节点 (局部索引)
            int u_global_idx = result.U_local_to_global_idx[u_local_idx]; // 获取其全局索引
            get_neighbors_optimized(u_global_idx, p, neighbor_cache); // 获取该节点的邻居 (全局索引)

            // 预分配邻接列表的空间，最大邻居数为 2*d
            if (result.final_adj_U_to_V_local[u_local_idx].capacity() < 2 * p.d) {
                 result.final_adj_U_to_V_local[u_local_idx].reserve(2 * p.d);
            }

            for (int neighbor_global_idx : neighbor_cache) { // 遍历所有邻居
                if (lattice[neighbor_global_idx] == 1) { // 如果邻居也是活动节点
                    int v_local_idx = global_to_V_local_map[neighbor_global_idx]; // 获取邻居在V中的局部索引
                    if (v_local_idx != -1) { // 如果邻居确实属于V集合 (即其坐标和为奇数)
                        // 添加一条从 u_local_idx 到 v_local_idx 的边
                        result.final_adj_U_to_V_local[u_local_idx].push_back(v_local_idx);
                    }
                }
            }
            
            // 对每个U节点的邻居列表进行排序和去重 (如果需要，但通常get_neighbors_optimized已处理重复)
            // 这里的去重主要防止因图结构导致的重复边记录，尽管物理邻居是唯一的。
            if(!result.final_adj_U_to_V_local[u_local_idx].empty()){
                std::sort(result.final_adj_U_to_V_local[u_local_idx].begin(), result.final_adj_U_to_V_local[u_local_idx].end());
                result.final_adj_U_to_V_local[u_local_idx].erase(
                    std::unique(result.final_adj_U_to_V_local[u_local_idx].begin(), result.final_adj_U_to_V_local[u_local_idx].end()),
                    result.final_adj_U_to_V_local[u_local_idx].end()
                );
            }
        }
    }

    result.mcm_size = 0; // 初始化最大匹配大小
    if (result.nU > 0 && result.nV > 0) { // 如果U和V都非空，可以运行Hopcroft-Karp
        HopcroftKarp::HKResult hk_res = HopcroftKarp::run(result.nU, result.nV, result.final_adj_U_to_V_local);
        result.mcm_size = hk_res.matching_size; // 存储最大匹配大小
        // 根据最大匹配计算MIS
        result.mis_nodes_global_indices = get_mis_nodes_indices(
            result.nU, result.nV,
            result.final_adj_U_to_V_local,
            hk_res.match_U,
            hk_res.match_V,
            result.U_local_to_global_idx,
            result.V_local_to_global_idx
        );
    } else { // 如果U或V为空 (或两者都为空)
        // MIS就是所有活动节点 (因为没有边)
        if (result.nU > 0) { // 如果只有U中有节点
            result.mis_nodes_global_indices = result.U_local_to_global_idx; 
        } else if (result.nV > 0) { // 如果只有V中有节点
            result.mis_nodes_global_indices = result.V_local_to_global_idx; 
        }
        // 如果U和V都为空，mis_nodes_global_indices 保持为空
    }

    // 根据计算出的MIS节点列表，更新MIS的晶格表示
    for (int mis_node_idx : result.mis_nodes_global_indices) {
        if (mis_node_idx >= 0 && mis_node_idx < total_nodes) { // 确保索引有效
            result.mis_lattice_representation[mis_node_idx] = 1; // 标记为MIS的一部分
        }
    }
    return result; // 返回包含MIS信息的结构体
}


// 存储样本基本度量值的结构体
struct SampleMetrics { 
    double E; // E度量值: MIS中节点数 / 活动子图中节点数
    double R; // R度量值: (MIS中U部分节点比例) - (MIS中V部分节点比例)
};

// 计算基本度量值E和R
// mis_result: MIS计算结果
// p: 参数结构体
// total_nodes: 晶格总节点数 (此处未使用，但可能用于未来扩展)
SampleMetrics calculate_metrics(const MisResult& mis_result, const Params& p, long long total_nodes) { 
    double E_val = 0.0; // 初始化E值
    double R_val = 0.0; // 初始化R值

    long long sum_mis = mis_result.mis_nodes_global_indices.size(); // MIS中的节点总数
    long long num_nodes_in_active_graph = static_cast<long long>(mis_result.nU) + mis_result.nV; // 活动子图中的节点总数 (U和V的大小之和)

    if (num_nodes_in_active_graph > 0) { // 避免除以零
        E_val = static_cast<double>(sum_mis) / num_nodes_in_active_graph; // E = |MIS| / |V_active|
    } else {
        E_val = 0.0; // 如果没有活动节点，E为0 (或可定义为1，取决于约定)
    }

    if (num_nodes_in_active_graph > 0) { // 只有活动节点存在时R才有意义
        long long mis_nodes_in_U_partition = 0; // MIS中属于U划分的节点数
        long long mis_nodes_in_V_partition = 0; // MIS中属于V划分的节点数

        // 遍历MIS中的每个节点，判断其属于U还是V划分
        for (int mis_node_global_idx : mis_result.mis_nodes_global_indices) {
            // 重新计算坐标和以确定划分 (与calculate_mis_representation中逻辑相同)
            int coord_sum_optimized = 0;
            long long temp_idx_for_sum = mis_node_global_idx;
             if (p.d == 0) {
                // 0维，坐标和为0 (偶数)
            } else if (p.d == 1) {
                coord_sum_optimized = temp_idx_for_sum % p.L;
            } else { 
                coord_sum_optimized += temp_idx_for_sum % p.W;
                temp_idx_for_sum /= p.W;
                for (int k_dim = p.d - 2; k_dim >= 1; --k_dim) {
                    coord_sum_optimized += temp_idx_for_sum % p.W;
                    temp_idx_for_sum /= p.W;
                }
                coord_sum_optimized += temp_idx_for_sum % p.L;
            }

            if (coord_sum_optimized % 2 == 0) { // 坐标和为偶数 -> U划分
                mis_nodes_in_U_partition++;
            } else { // 坐标和为奇数 -> V划分
                mis_nodes_in_V_partition++;
            }
        }

        // 计算MIS节点在U和V划分中的比例
        double prop_mis_in_U = (mis_result.nU > 0) ? static_cast<double>(mis_nodes_in_U_partition) / mis_result.nU : 0.0;
        double prop_mis_in_V = (mis_result.nV > 0) ? static_cast<double>(mis_nodes_in_V_partition) / mis_result.nV : 0.0;
        R_val = prop_mis_in_U - prop_mis_in_V; // R = (MIS_U / |U|) - (MIS_V / |V|)
    } else {
        R_val = 0.0; // 如果没有活动节点，R为0
    }
    return {E_val, R_val}; // 返回计算得到的E和R值
}


// 存储扩展度量值的结构体
struct ExtendedMetrics { 
    double E; // E度量值 (同SampleMetrics)
    double R; // R度量值 (同SampleMetrics)
    double bond_energy; // 键能 (平均的 s_i * s_j)
    double bond_defect; // 键缺陷 (平均的 (1-mis_i)*(1-mis_j))
};

// 计算扩展度量值 (包括E, R, 键能, 键缺陷)
// mis_result: MIS计算结果
// lattice: 当前晶格状态
// p: 参数结构体
// total_nodes: 晶格总节点数
// neighbor_cache: 邻居计算缓存
ExtendedMetrics calculate_extended_metrics( 
    const MisResult& mis_result,
    const std::vector<int>& lattice,
    const Params& p,
    long long total_nodes,
    std::vector<int>& neighbor_cache 
) {
    SampleMetrics original_metrics = calculate_metrics(mis_result, p, total_nodes); // 首先计算基本的E和R

    double bond_energy_sum = 0.0; // 键能总和
    double bond_defect_sum = 0.0; // 键缺陷总和
    long long bond_count_for_norm = 0; // 用于归一化的键总数 (活动节点间的边数)

    long long active_node_count = 0; // 活动节点总数 (值为1的节点)

    // 遍历所有节点，计算键相关的量
    for (long long i = 0; i < total_nodes; ++i) {
        if (lattice[i] == 1) { // 如果节点i是活动节点
            active_node_count++; // 活动节点计数增加
            get_neighbors_optimized(i, p, neighbor_cache); // 获取节点i的邻居
            for (int j : neighbor_cache) { // 遍历邻居j
                if (j > i && lattice[j] == 1) { // 如果邻居j也是活动节点，并且 j > i (避免重复计算每条边)
                    bond_count_for_norm++; // 键计数增加
                    int mis_i = mis_result.mis_lattice_representation[i]; // 节点i是否在MIS中 (0或1)
                    int mis_j = mis_result.mis_lattice_representation[j]; // 节点j是否在MIS中 (0或1)

                    // s_k = 2*mis_k - 1，使得在MIS中为+1，不在为-1
                    bond_energy_sum += (2.0 * mis_i - 1.0) * (2.0 * mis_j - 1.0); // 键能项 s_i * s_j
                    // (1-mis_i)*(1-mis_j) 项，当i和j都不在MIS中时为1，否则为0
                    bond_defect_sum += (1.0 - mis_i) * (1.0 - mis_j); // 键缺陷项
                }
            }
        }
    }

    double final_bond_energy = 0.0; // 最终的平均键能
    double final_bond_defect = 0.0; // 最终的平均键缺陷

    // 归一化：除以活动节点数 (原文分母是 N_active，不是 bond_count_for_norm)
    // 这表示每个活动节点对其邻居键的平均贡献。
    // 如果定义为每条边的平均值，则应除以 bond_count_for_norm。
    // 当前代码是除以 active_node_count。
    if (active_node_count > 0) { // 避免除以零
        final_bond_energy = bond_energy_sum / static_cast<double>(active_node_count);
        final_bond_defect = bond_defect_sum / static_cast<double>(active_node_count);
    }


    return {original_metrics.E, original_metrics.R, final_bond_energy, final_bond_defect}; // 返回所有扩展度量值
}


// 计算数据样本的绝对值一阶矩 (即平均绝对值)
// data: 输入的数据向量
double calculate_abs_first_moment(const std::vector<double>& data) {
    if (data.empty()) return 0.0; // 空数据集返回0
    double sum_abs = 0.0; // 绝对值总和
    for (double val : data) sum_abs += std::abs(val); // 累加每个数据的绝对值
    return sum_abs / data.size(); // 返回平均绝对值
}

// 计算数据样本的均值和标准差
// data: 输入的数据向量
// 返回值: pair<均值, 标准差>
std::pair<double, double> calculate_mean_stddev(const std::vector<double>& data) {
    if (data.empty()) return {0.0, 0.0}; // 空数据集返回0均值0标准差
    double sum = 0.0; // 数据总和
    double sum_sq = 0.0; // 数据平方和
    for (double val : data) {
        sum += val;
        sum_sq += val * val;
    }
    double mean = sum / data.size(); // 计算均值
    double variance = (sum_sq / data.size()) - (mean * mean); // 计算方差 E[X^2] - (E[X])^2
    // 处理由于浮点精度导致的微小负方差
    if (variance < 0 && std::abs(variance) < 1e-9) variance = 0.0; 
    else if (variance < 0) variance = 0.0; // 如果负方差较大，也强制为0 (可能表示数据问题)
    return {mean, std::sqrt(variance)}; // 返回均值和标准差 (标准差是方差的平方根)
}

// 计算数据样本的一阶、二阶和四阶矩
// data: 输入的数据向量
// 返回值: tuple<一阶矩, 二阶矩, 四阶矩>
std::tuple<double, double, double> calculate_moments(const std::vector<double>& data) {
    if (data.empty()) return {0.0, 0.0, 0.0}; // 空数据集返回全0矩
    double sum = 0.0; // E[X] 的累加部分
    double sum_sq = 0.0; // E[X^2] 的累加部分
    double sum_fourth = 0.0; // E[X^4] 的累加部分
    for (double val : data) {
        sum += val;
        sum_sq += val * val;
        sum_fourth += val * val * val * val;
    }
    size_t n = data.size(); // 样本数量
    if (n == 0) return {0.0, 0.0, 0.0}; // 再次检查样本数量 (理论上不会到这里如果上面已检查)
    return {sum / n, sum_sq / n, sum_fourth / n}; // 返回计算的各阶矩
}



//  g++ -O3 -fopenmp -march=native MIS-multi-L-v11-6.cpp -o MIS;./MIS

// g++ -O3 -march=native -mtune=native -fopenmp -funroll-loops -ffast-math -DNDEBUG MIS-multi-L-v11-5.cpp -o MIS;./MIS

// 主函数入口
int main(int argc, char* argv[]) {
    // 优化C++标准流的性能，解除与C标准IO的同步，并取消cin/cout的绑定
    std::ios_base::sync_with_stdio(false); 
    std::cin.tie(NULL);
    std::cout.tie(NULL);

    // 输出当前环境可用的最大线程数
    std::cout << "Max thread available: " << omp_get_max_threads() << std::endl;

    Params p_template; // 创建参数模板对象
    // 设置默认参数
    p_template.d = 2; // 默认维度
    p_template.save_state = false; // 默认不保存状态
    p_template.base_seed = 0; // 默认基础种子 (0表示使用随机设备生成)
    p_template.periodic_boundary = true; // 默认使用周期性边界
    int num_samples_per_group = 1000; // 每组参数组合的样本数

    // 从命令行参数解析并覆盖默认参数
    if (argc > 1) p_template.d = std::stoi(argv[1]); // 第1个参数: 维度d
    if (argc > 2) p_template.save_state = (std::stoi(argv[2]) == 1); // 第2个参数: 是否保存状态 (1为是)
    if (argc > 3) { // 第3个参数: 基础种子
        try {
            long long temp_seed = std::stoll(argv[3]); // 使用long long读取，以检查范围
            if (temp_seed == 0) { // 如果输入为0，则遵循默认行为 (后续用random_device)
                 p_template.base_seed = 0;
            } else if (temp_seed < 0 || temp_seed > std::numeric_limits<unsigned int>::max()) { // 超出unsigned int范围
                std::cerr << "Warning: Provided seed " << temp_seed << " is out of range for unsigned int. Using random_device." << std::endl;
                p_template.base_seed = 0; // 回退到使用random_device
            } else {
                p_template.base_seed = static_cast<unsigned int>(temp_seed); // 合法种子
            }
        } catch (const std::exception& e) { // 解析异常
            std::cerr << "Warning: Invalid seed value '" << argv[3] << "'. Using random_device. Error: " << e.what() << std::endl;
            p_template.base_seed = 0; // 回退到使用random_device
        }
    }
    if (argc > 4) p_template.periodic_boundary = (std::stoi(argv[4]) == 1); // 第4个参数: 是否周期边界 (1为是)
    if (argc > 5) num_samples_per_group = std::stoi(argv[5]); // 第5个参数: 每组样本数

    // 输出最终使用的全局参数
    std::cout << "Global parameters: d=" << p_template.d
              << ", save_combined_state_per_sample=" << p_template.save_state
              << ", base_seed=" << p_template.base_seed
              << ", periodic_boundary=" << p_template.periodic_boundary
              << ", samples_per_group=" << num_samples_per_group
              << std::endl;

    // 处理基础种子：如果为0，则使用随机设备生成一个实际的初始种子
    unsigned int actual_initial_seed = p_template.base_seed;
    if (p_template.base_seed == 0) {
        std::random_device rd; // 创建随机设备
        actual_initial_seed = rd(); // 生成随机种子
        std::cout << "Using random_device to generate initial seed: " << actual_initial_seed << " (input base_seed was 0)" << std::endl;
    } else {
        std::cout << "Using provided initial seed: " << actual_initial_seed << std::endl;
    }
    p_template.base_seed = actual_initial_seed; // 更新模板中的种子为实际使用的初始种子


    std::vector<int> L_list; // 存储L值的列表 (系统尺寸)
    std::vector<double> rho_list; // 存储rho值的列表 (密度)

    // 根据维度d设置L和rho的测试列表
    if (p_template.d == 2){ // 2维情况
        L_list = {400,800,1200,1600,2000,2400,2800,3200,3600,4000}; // L值列表
        // rho值列表 (较密集，覆盖特定相变区域)
        rho_list = {0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.089, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.1, 0.101, 0.102, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.11, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116, 0.117, 0.118, 0.119, 0.12, 0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129, 0.13};
    } else if (p_template.d == 3) { // 3维情况
        L_list =  {40,60,80,100,120,140,160,180,200,220}; // L值列表
        // rho值列表 (较密集)
        rho_list = {0.13, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139, 0.14, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149, 0.15, 0.151, 0.152, 0.153, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.16, 0.161, 0.162, 0.163, 0.164, 0.165, 0.166, 0.167, 0.168, 0.169, 0.17, 0.171, 0.172, 0.173, 0.174, 0.175, 0.176, 0.177, 0.178, 0.179}; // 密度列表}

        // rho_list = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.102, 0.104, 0.106, 0.108, 0.11, 0.112, 0.114, 0.116, 0.118, 0.12, 0.122, 0.124, 0.126, 0.128, 0.13, 0.132, 0.134, 0.136, 0.138, 0.14, 0.142, 0.144, 0.146, 0.148, 0.15, 0.152, 0.154, 0.156, 0.158, 0.16, 0.162, 0.164, 0.166, 0.168, 0.17, 0.172, 0.174, 0.176, 0.178, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26};
    }

    // 设置结果文件存储目录
    std::string results_dir = "MIS-graph/MIS-data-multi-0625"; 
    #if __has_include(<filesystem>) // 检查并创建目录 (同上)
        if (!fs::exists(results_dir)) {
            fs::create_directories(results_dir);
        }
    #else
        create_directories_recursive(results_dir);
    #endif

    // 构建各度量值输出文件的完整路径
    std::string e_filename = results_dir + "/E_values_all-d-" + std::to_string(p_template.d) + ".txt";
    std::string r_filename = results_dir + "/R_values_all-d-" + std::to_string(p_template.d) + ".txt";
    std::string bond_energy_filename = results_dir + "/bond_energy_values_all-d-" + std::to_string(p_template.d) + ".txt";
    std::string bond_defect_filename = results_dir + "/bond_defect_values_all-d-" + std::to_string(p_template.d) + ".txt";
    std::string summary_filename = results_dir + "/summary_stats-d-" + std::to_string(p_template.d) + ".txt";

    // 打开所有输出文件流
    std::ofstream e_out(e_filename);
    std::ofstream r_out(r_filename);
    std::ofstream bond_energy_out(bond_energy_filename);
    std::ofstream bond_defect_out(bond_defect_filename);
    std::ofstream summary_out(summary_filename);

    // 检查文件是否成功打开
    if (!e_out.is_open() || !r_out.is_open() || !bond_energy_out.is_open() ||
        !bond_defect_out.is_open()  || !summary_out.is_open()) {
        std::cerr << "Error: Could not open one or more output files in " << results_dir << std::endl;
        return 1; // 打开失败，程序退出
    }

    // 写入汇总统计文件的表头
    summary_out << "L,rho,E_1st,E_2nd,E_4th,abs_R_1st,R_1st,R_2nd,R_4th,bond_energy_1st,bond_energy_2nd,bond_energy_4th,bond_defect_1st,bond_defect_2nd,bond_defect_4th" << std::endl;

    // 主循环：遍历所有L值和rho值的组合
    for (int current_L : L_list) {
        for (double current_rho : rho_list) {
            Params p = p_template; // 基于模板创建当前参数p
            p.L = current_L; // 设置当前L
            p.W = current_L; // 设置当前W (假设W=L, 即超立方体)
            p.rho = current_rho; // 设置当前rho

            std::cout << "\nProcessing: L=" << p.L << ", W=" << p.W << ", rho=" << format_rho_string(p.rho)
                      << ", d=" << p.d << std::endl; // 输出当前处理的参数组合

            long long total_nodes = 1; // 初始化总节点数
            try { // 尝试计算总节点数，捕获可能的溢出或错误
                if (p.d < 0 || p.L < 0 || p.W < 0) throw std::runtime_error("d, L and W must be non-negative"); // 参数有效性检查
                if (p.L == 0 && p.d >= 1) total_nodes = 0; // L=0且d>=1 -> 0节点
                else if (p.W == 0 && p.d >= 2) total_nodes = 0; // W=0且d>=2 -> 0节点
                else if (p.d == 0) total_nodes = 1; // 0维系统有1个节点
                else if (p.d == 1) total_nodes = p.L; // 1维系统有L个节点
                else { // d > 1
                    total_nodes = p.L; // 初始为L (第0维)
                    for(int i_dim = 1; i_dim < p.d; ++i_dim) { // 累乘其余d-1个维度
                        if (p.W == 0) { total_nodes = 0; break; } // 如果W=0，总节点数为0
                        // 使用 __int128 防止中间乘法溢出 long long
                        unsigned __int128 temp_total_nodes = static_cast<unsigned __int128>(total_nodes) * p.W;
                        if (temp_total_nodes > std::numeric_limits<long long>::max()) { // 检查是否溢出 long long
                             throw std::overflow_error("L * W^(d-1) too large for long long");
                        }
                        total_nodes = static_cast<long long>(temp_total_nodes); // 更新总节点数
                    }
                }
                if (total_nodes < 0) { // 理论上不会发生，除非上面逻辑有误或参数极端
                     throw std::runtime_error("Invalid L/W/d combination results in negative total_nodes");
                }
            } catch (const std::overflow_error& e) { // 捕获溢出错误
                std::cerr << "Error calculating total_nodes for (L,W,d)=(" << p.L << "," << p.W << "," << p.d << "): " << e.what() << std::endl;
                continue; // 跳过当前参数组合
            } catch (const std::runtime_error& e) { // 捕获其他运行时错误
                std::cerr << "Error for (L,W,d)=(" << p.L << "," << p.W << "," << p.d << "): " << e.what() << std::endl;
                continue; // 跳过当前参数组合
            }

            if (total_nodes == 0) { // 如果总节点数为0
                std::cout << "  Total nodes is 0. Skipping sampling." << std::endl;
                // 写入空结果到各文件
                e_out << p.L << "," << format_rho_string(p.rho);
                r_out << p.L << "," << format_rho_string(p.rho);
                bond_energy_out << p.L << "," << format_rho_string(p.rho);
                bond_defect_out << p.L << "," << format_rho_string(p.rho);

                for(int i=0; i<num_samples_per_group; ++i) { // 对每个样本写入0
                    e_out << "," << 0.0;
                    r_out << "," << 0.0;
                    bond_energy_out << "," << 0.0;
                    bond_defect_out << "," << 0.0;
                }
                e_out << std::endl;
                r_out << std::endl;
                bond_energy_out << std::endl;
                bond_defect_out << std::endl;
                summary_out << p.L << "," << format_rho_string(p.rho) << ",0,0,0,0,0,0,0,0,0,0,0,0,0" << std::endl; 
                continue; // 跳过采样
            }

            // 计算 L_multipliers: 用于索引转换的乘数
            // L_multipliers[i] 是第 i 维坐标在将多维坐标转换为一维索引时的权重/乘数。
            // 例如，对于3D (c0, c1, c2) 和尺寸 L, W, W:
            // index = c0*W*W + c1*W + c2*1
            // L_multipliers[0] = W*W
            // L_multipliers[1] = W
            // L_multipliers[2] = 1
            p.L_multipliers.assign(p.d, 1LL); // 初始化所有乘数为1 (long long)
            if (p.d > 0 && total_nodes > 0) { // 仅当维度大于0且节点数大于0时计算
                if (p.d == 1) { // 1维情况
                    p.L_multipliers[0] = 1LL; // 乘数为1
                } else { // d > 1 情况
                    // p.L_multipliers[p.d-1] 已经是 1LL (最低维的乘数)
                    // 从 d-2 维倒推到 1 维
                    for (int k = p.d - 2; k >= 1; --k) { 
                        // L_multipliers[k] = L_multipliers[k+1] * W
                        unsigned __int128 temp_mult = static_cast<unsigned __int128>(p.L_multipliers[k+1]) * p.W;
                         if (temp_mult > std::numeric_limits<long long>::max()){ // 检查溢出
                             std::cerr << "Warning: L_multiplier overflow for W=" << p.W << ", d=" << p.d << ". Results may be incorrect." << std::endl;
                             p.L_multipliers[k] = std::numeric_limits<long long>::max(); // 设为最大值以示错误
                        } else {
                           p.L_multipliers[k] = static_cast<long long>(temp_mult);
                        }
                    }
                    
                    // 计算第 0 维的乘数: L_multipliers[0] = L_multipliers[1] * W
                    // (注意：这里假设第0维的“右边”是第1维，其大小为W。如果第0维是L，第1维是W1, 第2维是W2...
                    // 那么 c0 * (W1*W2*...) + c1 * (W2*W3*...) + ...
                    // 当前代码的 coords_to_index 和 L_multipliers 计算是一致的，都假设第0维是L，其余是W)
                    unsigned __int128 temp_mult_dim0 = static_cast<unsigned __int128>(p.L_multipliers[1]) * p.W;
                     if (temp_mult_dim0 > std::numeric_limits<long long>::max()){ // 检查溢出
                        std::cerr << "Warning: L_multiplier overflow for first dimension. Results may be incorrect." << std::endl;
                        p.L_multipliers[0] = std::numeric_limits<long long>::max();
                    } else {
                       p.L_multipliers[0] = static_cast<long long>(temp_mult_dim0);
                    }
                }
            }

            // 为每个样本存储度量值的向量
            std::vector<double> E_samples(num_samples_per_group);
            std::vector<double> R_samples(num_samples_per_group);
            std::vector<double> bond_energy_samples(num_samples_per_group);
            std::vector<double> bond_defect_samples(num_samples_per_group);

            auto group_time_start = std::chrono::high_resolution_clock::now(); // 记录当前参数组开始时间

            // 使用OpenMP并行处理样本
            #pragma omp parallel
            {
                // 每个线程有自己的邻居缓存，避免竞争
                std::vector<int> local_neighbor_cache;
                if (p.d > 0) local_neighbor_cache.reserve(2 * p.d); // 预分配容量 (最大邻居数 2*d)

                // OpenMP并行for循环，将样本分配给不同线程
                #pragma omp for
                for (int i = 0; i < num_samples_per_group; ++i) {
                    Params p_local = p; // 复制一份参数供当前样本使用 (主要是为了种子独立)
                    unsigned int sample_seed = p.base_seed + i; // 为每个样本生成独立的种子
                    std::mt19937 gen(sample_seed); // 使用样本种子初始化随机数生成器

                    // 模拟过程
                    std::vector<int> lattice = generate_initial_lattice(total_nodes, p_local.rho, gen); // 1. 生成初始晶格
                    apply_rules(lattice, p_local, total_nodes, local_neighbor_cache); // 2. 应用规则演化晶格
                    MisResult mis_res = calculate_mis_representation(lattice, p_local, total_nodes, local_neighbor_cache); // 3. 计算MIS
                    ExtendedMetrics metrics = calculate_extended_metrics(mis_res, lattice, p_local, total_nodes, local_neighbor_cache); // 4. 计算度量值

                    // 存储当前样本的度量值
                    E_samples[i] = metrics.E;
                    R_samples[i] = metrics.R;
                    bond_energy_samples[i] = metrics.bond_energy;
                    bond_defect_samples[i] = metrics.bond_defect;

                    // 如果设置了保存状态的标志
                    if (p_local.save_state) {
                        // 使用OpenMP临界区确保文件写入操作的原子性/互斥性
                        #pragma omp critical (save_file_section) 
                        { 
                           save_state_to_file(lattice, mis_res.mis_lattice_representation, p_local, sample_seed); // 保存状态到文件
                        }
                    }

                    // 进度输出 (由0号线程每一千个样本输出一次，或按总样本数的10%输出)
                    // 注意: i % 1000 在并行循环中可能不按预期工作，因为i是全局迭代变量，
                    // 线程可能处理不连续的i块。但这里仅用于粗略进度指示。
                    // if (omp_get_thread_num() == 0 && i % 1000 == 0 ) { 
                    //     // 更精细的输出控制：大约每10%输出一次，或在最后一个样本时输出
                    //     if ((i + 1) % (num_samples_per_group / 10 == 0 ? 1 : num_samples_per_group / 10) == 0 || i == num_samples_per_group - 1) {
                    //          if (num_samples_per_group >=10 && (i+1) % (num_samples_per_group/10) == 0 ) // 如果总样本数大于等于10，则每10%输出
                    //             std::cout << "\r  (L=" << p_local.L << ", rho=" << format_rho_string(p_local.rho) << ") Sample " << i + 1 << "/" << num_samples_per_group << " done." << std::flush;
                    //          else if (i == num_samples_per_group -1) // 或者在最后一个样本完成时输出
                    //             std::cout << "\r  (L=" << p_local.L << ", rho=" << format_rho_string(p_local.rho) << ") Sample " << i + 1 << "/" << num_samples_per_group << " done." << std::flush;
                    //     }
                    // }
                } // 结束并行for循环
            } // 结束并行区域
            std::cout << std::endl; // 确保进度条后的输出换行

            auto group_time_end = std::chrono::high_resolution_clock::now(); // 记录当前参数组结束时间
            std::chrono::duration<double> group_duration = group_time_end - group_time_start; // 计算耗时
            std::cout << "  Finished " << num_samples_per_group << " samples in " << std::fixed << std::setprecision(2) << group_duration.count() << "s." << std::endl; // 输出耗时

            // 将当前参数组的所有样本数据写入各自的文件
            // E值
            e_out << p.L << "," << format_rho_string(p.rho); // 写入L和rho
            for (double val : E_samples) e_out << "," << std::fixed << std::setprecision(8) << val; // 写入每个E样本值
            e_out << std::endl;

            // R值
            r_out << p.L << "," << format_rho_string(p.rho);
            for (double val : R_samples) r_out << "," << std::fixed << std::setprecision(8) << val;
            r_out << std::endl;

            // 键能值
            bond_energy_out << p.L << "," << format_rho_string(p.rho);
            for (double val : bond_energy_samples) bond_energy_out << "," << std::fixed << std::setprecision(8) << val;
            bond_energy_out << std::endl;

            // 键缺陷值
            bond_defect_out << p.L << "," << format_rho_string(p.rho);
            for (double val : bond_defect_samples) bond_defect_out << "," << std::fixed << std::setprecision(8) << val;
            bond_defect_out << std::endl;

            // 计算并写入汇总统计数据 (各阶矩)
            std::tuple<double, double, double> E_moments = calculate_moments(E_samples);
            std::tuple<double, double, double> R_moments = calculate_moments(R_samples);
            std::tuple<double, double, double> bond_energy_moments = calculate_moments(bond_energy_samples);
            std::tuple<double, double, double> bond_defect_moments = calculate_moments(bond_defect_samples);
            double abs_R_first_moment = calculate_abs_first_moment(R_samples); // R的绝对值一阶矩

            summary_out << p.L << "," << format_rho_string(p.rho) << ","
                        << std::fixed << std::setprecision(8) << std::get<0>(E_moments) << "," // E一阶矩 (均值)
                        << std::fixed << std::setprecision(8) << std::get<1>(E_moments) << "," // E二阶矩
                        << std::fixed << std::setprecision(8) << std::get<2>(E_moments) << "," // E四阶矩
                        << std::fixed << std::setprecision(8) << abs_R_first_moment << ","    // |R|一阶矩
                        << std::fixed << std::setprecision(8) << std::get<0>(R_moments) << "," // R一阶矩 (均值)
                        << std::fixed << std::setprecision(8) << std::get<1>(R_moments) << "," // R二阶矩
                        << std::fixed << std::setprecision(8) << std::get<2>(R_moments) << "," // R四阶矩
                        << std::fixed << std::setprecision(8) << std::get<0>(bond_energy_moments) << "," // 键能一阶矩
                        << std::fixed << std::setprecision(8) << std::get<1>(bond_energy_moments) << "," // 键能二阶矩
                        << std::fixed << std::setprecision(8) << std::get<2>(bond_energy_moments) << "," // 键能四阶矩
                        << std::fixed << std::setprecision(8) << std::get<0>(bond_defect_moments) << "," // 键缺陷一阶矩
                        << std::fixed << std::setprecision(8) << std::get<1>(bond_defect_moments) << "," // 键缺陷二阶矩
                        << std::fixed << std::setprecision(8) << std::get<2>(bond_defect_moments)       // 键缺陷四阶矩
                        << std::endl;

            // 输出当前参数组的简要统计结果到控制台
            std::cout << "  Stats: E_mean=" << std::get<0>(E_moments)<< ", R_mean=" << std::get<0>(R_moments) << std::endl;
        } // 结束rho循环
    } // 结束L循环

    // 关闭所有输出文件
    e_out.close();
    r_out.close();
    summary_out.close();
    bond_energy_out.close();
    bond_defect_out.close();

    // 输出最终完成信息和结果文件路径
    std::cout << "\nAll processing finished. Results saved." << std::endl;
    std::cout << "E values: " << e_filename << std::endl;
    std::cout << "R values: " << r_filename << std::endl;
    std::cout << "Summary: " << summary_filename << std::endl;
    std::cout << "Bond energy values: " << bond_energy_filename << std::endl;
    std::cout << "Bond defect values: " << bond_defect_filename << std::endl;

    if (p_template.save_state) {
        std::cout << "Combined states (if any saved) are in MIS-graph/mis-data/" << std::endl;
    }

    return 0;
}

