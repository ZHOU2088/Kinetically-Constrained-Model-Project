#include <iostream> // 导入输入输出流库，用于控制台输入输出
#include <vector> // 导入向量容器库，用于动态数组
#include <numeric> // 导入数值算法库，例如 accumulate (求和)
#include <algorithm> // 导入算法库，例如 sort (排序), unique (去重)
#include <random> // 导入随机数生成库
#include <cmath> // 导入数学函数库，例如 abs (绝对值), sqrt (平方根), pow (幂运算)
#include <fstream> // 导入文件流库，用于文件读写
#include <string> // 导入字符串处理库
#include <map>
#include <sstream> // 导入字符串流库，用于字符串和流的转换
#include <iomanip> // 导入输入输出操纵符库，例如 setw (设置宽度), setprecision (设置精度)
#include <chrono> // 导入时间库，用于计时
// #include <map> // 导入映射容器库 (通常基于红黑树实现，键值对有序)
#include <set> // 导入集合容器库 (通常基于红黑树实现，元素有序且唯一)
#include <limits> // 导入数值极限库，例如 numeric_limits (获取类型的最大/最小值)
#include <unordered_map> // 导入无序映射容器库 (通常基于哈希表实现，键值对无序)
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#if __has_include(<filesystem>) // 预处理指令：检查是否包含 <filesystem> 头文件 (C++17及以后标准)
#include <filesystem> // 如果可用，导入文件系统库
namespace fs = std::filesystem; // 为 std::filesystem 创建命名空间别名 fs，方便使用
#else // 如果系统不直接支持 <filesystem> (例如旧版编译器或标准库)
#include <sys/stat.h> // 导入 POSIX 系统调用 stat.h，用于获取文件状态和创建目录 (mkdir)
#ifdef _WIN32 // 预处理指令：如果目标平台是 Windows
#include <direct.h> // 导入 Windows 特有的目录操作库 (包含 _mkdir)
#endif // _WIN32 结束
void create_directories_compat(const std::string& path) { // 定义一个兼容的创建目录函数
    #ifdef _WIN32 // 如果是 Windows 平台
        _mkdir(path.c_str()); // 使用 Windows 的 _mkdir 函数创建目录
    #else // 如果是其他 (通常是 POSIX 兼容) 平台
        mkdir(path.c_str(), 0755); // 使用 POSIX 的 mkdir 函数创建目录，权限设置为 0755 (用户读写执行，组和其他用户读执行)
    #endif // _WIN32 结束
}
#endif // __has_include(<filesystem>) 结束


struct Params { // 定义参数结构体 Params，用于封装程序运行所需的各种参数设置
    int d; // 晶格的维度 (dimension)
    int L; // 晶格在第一个维度上的长度 (Length)
    int W; // 晶格在其他维度上的宽度 (Width)
    double rho; // 一个概率参数，通常与初始节点状态有关 (例如，节点为0的概率)
    unsigned int seed; // 用于随机数生成器的种子，以确保实验的可复现性
    bool periodic_boundary; // 布尔标志，指示是否使用周期性边界条件
    std::vector<long long> L_multipliers; // 存储各维度乘数的向量，用于快速从坐标计算一维索引或邻居索引
    long long total_nodes_for_bitpacking; // 用于位打包的总节点数，即晶格中的总位置数
    int num_samples;
    char boundary_prefer;
}; // Params 结构体定义结束


inline long long get_num_blocks(long long total_bits) { // 内联函数 get_num_blocks：计算存储指定数量的位 (total_bits) 需要多少个64位无符号整数块
    if (total_bits == 0) return 0; // 如果总位数为0，则不需要任何块
    return (total_bits + 63) / 64; // 将总位数加上63再除以64，实现向上取整，确保所有位都能被存储
} // get_num_blocks 函数定义结束


inline int get_bit(const std::vector<uint64_t>& packed_vec, long long index) { // 内联函数 get_bit：从使用位打包的向量 (packed_vec) 中获取指定索引 (index) 处的位值 (0或1)
    // packed_vec: 存储打包位的 uint64_t 向量
    // index: 要获取的位的全局一维索引
    long long block_idx = index / 64; // 计算包含该位的块在向量中的索引 (每块64位)
    int bit_in_block = index % 64; // 计算该位在对应块内的偏移量 (0-63)
    return (packed_vec[block_idx] >> bit_in_block) & 1ULL; // 将对应块右移 bit_in_block 位，使目标位到达最低位，然后与1进行按位与操作，提取出该位的值
} // get_bit 函数定义结束


inline void set_bit(std::vector<uint64_t>& packed_vec, long long index, int value) { // 内联函数 set_bit：在位打包的向量 (packed_vec) 中设置指定索引 (index) 处的位值为指定值 (value)
    // packed_vec: 存储打包位的 uint64_t 向量 (可修改)
    // index: 要设置的位的全局一维索引
    // value: 要设置的值，应为0或1
    long long block_idx = index / 64; // 计算包含该位的块在向量中的索引
    int bit_in_block = index % 64; // 计算该位在对应块内的偏移量
    if (value == 1) { // 如果要将位设置为1
        packed_vec[block_idx] |= (1ULL << bit_in_block); // 使用按位或操作，将1左移 bit_in_block 位得到的掩码与块内容结合，将目标位置1
    } else { // 如果要将位设置为0
        packed_vec[block_idx] &= ~(1ULL << bit_in_block); // 使用按位与操作，将1左移 bit_in_block 位得到的掩码取反 (目标位为0，其余为1)，与块内容结合，将目标位置0
    }
} // set_bit 函数定义结束



std::vector<int> index_to_coords(int index, int L, int W, int d) { // 函数 index_to_coords：将一维线性索引转换为 d 维坐标
    std::vector<int> coords(d); // 创建一个大小为 d 的向量，用于存储计算出的各维度坐标
    if (d == 0) return coords; // 如果维度为0，直接返回空坐标向量 (特殊情况，通常表示单个点或无效)
    if (d == 1) { // 如果是一维情况
        if (L <= 0) return coords; // 如果长度 L 无效 (小于等于0)，返回空坐标 (或根据错误处理策略调整)
        coords[0] = index % L; // 一维坐标即为索引值 (对L取模可能用于周期性边界或确保在范围内)
        return coords; // 返回包含单个坐标的向量
    }
    // 对于 d > 1 的情况
    for (int i = d - 1; i >= 1; --i) { // 从最后一个维度 (d-1) 迭代到第二个维度 (1) (通常是宽度 W)
        coords[i] = index % W; // 当前维度的坐标是索引对 W 取模的结果
        index /= W; // 更新索引，除去当前已计算维度的影响 (整数除法)
    }
    coords[0] = index % L; // 第一个维度 (通常是长度 L) 的坐标是剩余索引对 L 取模的结果
    return coords; // 返回计算得到的 d 维坐标向量
} // index_to_coords 函数定义结束

int coords_to_index(const std::vector<int>& coords, int L, int W, int d) { // 函数 coords_to_index：将 d 维坐标转换为一维线性索引
    if (d == 0) return 0; // 如果维度为0，返回索引0 (或某个定义的特殊值)
    if (d == 1) { // 如果是一维情况
        if (L <= 0) return 0; // 如果长度 L 无效，返回索引0
        return coords[0] % L; // 一维索引即为坐标值 (对L取模以确保在有效范围内)
    }
    // 对于 d > 1 的情况
    int index = coords[0]; // 从第一个维度的坐标开始初始化索引
    for (int i = 1; i < d; ++i) { // 遍历从第二个维度到最后一个维度
        index = index * W + coords[i]; // 根据“行主序”或类似方式累积计算索引：(前缀索引 * 当前维度大小 + 当前坐标)
    }
    return index; // 返回计算得到的一维索引
} // coords_to_index 函数定义结束

std::vector<int> get_neighbors(int index, const Params& p) { // 函数 get_neighbors：获取给定一维索引 (index) 的节点在晶格中的所有邻居节点的索引列表
    std::vector<int> neighbors; // 创建一个向量用于存储邻居节点的索引
    if (p.d == 0 || p.L <= 0 || (p.d > 1 && p.W <= 0)) return neighbors; // 如果维度、长度或宽度无效，则没有邻居，返回空列表
    neighbors.reserve(2 * p.d); // 为邻居列表预留空间，每个维度最多有两个邻居 (正方向和负方向)
    
    std::vector<int> current_coords = index_to_coords(index, p.L, p.W, p.d); // 将当前节点的一维索引转换为 d 维坐标

    for (int i = 0; i < p.d; ++i) { // 遍历晶格的每一个维度
        int dim_size = (i == 0) ? p.L : p.W; // 获取当前维度的大小 (第一个维度是L，其余是W)
        if (dim_size <= 1) continue; // 如果当前维度大小为1或更小，则该维度上没有不同的邻居，跳过

        long long multiplier = p.L_multipliers[i]; // 获取当前维度的乘数，用于从坐标变化快速计算索引变化
        int original_coord_val = current_coords[i]; // 当前节点在当前维度上的坐标值

        if (p.periodic_boundary) { // 如果使用周期性边界条件
            // 计算正方向的邻居 (坐标值增加)
            if (original_coord_val == dim_size - 1) { // 如果节点在当前维度的上边界
                neighbors.push_back(index - original_coord_val * multiplier); // 邻居是该维度的起点 (通过索引减去整个维度的偏移)
            } else { // 如果节点不在上边界
                neighbors.push_back(index + multiplier); // 邻居索引是当前索引加上维度乘数
            }
            // 计算负方向的邻居 (坐标值减少)
            if (original_coord_val == 0) { // 如果节点在当前维度的下边界
                neighbors.push_back(index + (dim_size - 1) * multiplier); // 邻居是该维度的终点 (通过索引加上(维度大小-1)倍的乘数)
            } else { // 如果节点不在下边界
                neighbors.push_back(index - multiplier); // 邻居索引是当前索引减去维度乘数
            }
        } else { // 如果不使用周期性边界条件 (即固定边界或吸收边界)
            // 计算正方向的邻居
            if (original_coord_val + 1 < dim_size) { // 如果节点不在当前维度的上边界 (即 original_coord_val < dim_size - 1)
                neighbors.push_back(index + multiplier); // 添加正方向邻居
            }
            // 计算负方向的邻居
            if (original_coord_val > 0) { // 如果节点不在当前维度的下边界 (即 original_coord_val > 0)
                neighbors.push_back(index - multiplier); // 添加负方向邻居
            }
        }
    }

    // 特殊处理：当L或W为2且维度大于1时，可能因周期性边界等因素导致计算出的邻居索引重复
    // 例如，在L=2的维度上，从0号位置前进1和后退1（周期性）都到达1号位置。
    // 此处排序并移除重复项以确保邻居列表的唯一性。
    if ((p.L == 2 || p.W == 2) && p.d > 1) {
        std::sort(neighbors.begin(), neighbors.end()); // 对邻居索引列表进行排序
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end()); // 移除排序后列表中的重复元素
    }
    return neighbors; // 返回计算得到的邻居列表
} // get_neighbors 函数定义结束


std::vector<uint64_t> generate_initial_lattice_packed(long long total_nodes, double rho, std::mt19937& gen) { // 函数 generate_initial_lattice_packed：生成使用位打包的初始晶格状态
    long long num_blocks = get_num_blocks(total_nodes); // 计算存储 total_nodes 个状态需要多少个64位块
    std::vector<uint64_t> lattice_packed(num_blocks, 0ULL); // 初始化位打包晶格向量，所有位默认为0 (0ULL表示无符号长整型0)
    std::uniform_real_distribution<> distrib(0.0, 1.0); // 创建一个在 [0.0, 1.0) 区间内均匀分布的实数随机数生成器
    
    for (long long i = 0; i < total_nodes; ++i) { // 遍历晶格中的每一个节点位置
        if (distrib(gen) >= rho) { // 生成一个随机数，如果该随机数大于等于 rho (即以 1-rho 的概率)
            set_bit(lattice_packed, i, 1); // 将当前节点 i 的状态设置为1
        }
        // 如果随机数小于 rho (即以 rho 的概率)，则节点状态保持为0 (因为 lattice_packed 初始化时所有位为0)
    }
    return lattice_packed; // 返回生成的初始晶格状态 (位打包形式)
} // generate_initial_lattice_packed 函数定义结束



void apply_rules_packed(std::vector<uint64_t>& lattice_packed, const Params& p) { // 函数 apply_rules_packed：对位打包的晶格应用定义的细胞自动机演化规则
    if (p.total_nodes_for_bitpacking == 0) return; // 如果总节点数为0，则无需应用规则，直接返回

    std::vector<long long> nodes_to_become_one; // 创建一个向量，用于存储那些根据规则要从0变为1的节点的索引
    if (p.rho > 0.0) { // 这是一个启发式优化：如果 rho > 0 (意味着初始可能存在0状态的节点)
        // 预估可能变为1的节点数量，并为此向量预分配内存，以提高效率
        // (1.0 - p.rho) 是初始时1的比例的估计，再乘以一个小的因子 (0.1) 作为从0变1的节点比例的粗略估计
        nodes_to_become_one.reserve(static_cast<size_t>(p.total_nodes_for_bitpacking * (1.0 - p.rho) * 0.1));
    }

    // 规则1 (暂定): 如果一个节点为0，且其所有邻居都为1，则该节点在下一状态变为1。
    // 此规则的更新是基于当前时刻的晶格状态，变更将在第一轮扫描后统一应用。
    for (long long i = 0; i < p.total_nodes_for_bitpacking; ++i) { // 遍历所有节点
        if (get_bit(lattice_packed, i) == 0) { // 如果当前节点 i 的状态是0
            std::vector<int> neighbors = get_neighbors(i, p); // 获取节点 i 的所有邻居
            if (neighbors.empty() && p.d > 0) { // 如果节点没有邻居 (且维度 > 0，对于非0维孤立点)
                // 对于没有邻居的0状态节点，是否变为1取决于规则定义。当前逻辑是跳过，即保持0。
                // 如果规则是“所有邻居都为1”被真空地满足（vacuously true），则应变为1。
                continue; // 跳过此节点
            }
            if (neighbors.empty() && p.d == 0) { // 特殊处理0维情况 (单个节点)
                 // 如果是0维，单个节点若为0，没有邻居，按当前逻辑也会跳过，保持0。
                 continue; // 跳过
            }

            bool all_neighbors_one = true; // 初始化标志，假设所有邻居都为1
            for (int neighbor_idx : neighbors) { // 遍历所有邻居
                if (get_bit(lattice_packed, neighbor_idx) == 0) { // 如果发现任何一个邻居的状态是0
                    all_neighbors_one = false; // 则标志置为false
                    break; // 无需再检查其他邻居，跳出内部循环
                }
            }
            if (all_neighbors_one) { // 如果所有邻居的状态确实都为1 (或者没有邻居且规则如此定义)
                nodes_to_become_one.push_back(i); // 将节点 i 的索引加入待变为1的列表
            }
        }
    }
    // 应用规则1的变更：将所有被标记的0状态节点变为1状态节点
    for (long long node_idx : nodes_to_become_one) { // 遍历待变为1的节点列表
        set_bit(lattice_packed, node_idx, 1); // 将这些节点的状态设置为1
    }

    // 规则2 (暂定): 如果一个节点为0，则其所有邻居都变为0。
    // 此规则在规则1应用更新后的晶格上执行。
    // 使用一个临时标记数组 to_be_zeroed 来记录哪些节点需要变为0，避免在迭代过程中修改晶格状态影响后续判断。
    std::vector<bool> to_be_zeroed(p.total_nodes_for_bitpacking, false); // 初始化标记数组，所有元素为false
    for (long long i = 0; i < p.total_nodes_for_bitpacking; ++i) { // 再次遍历所有节点 (基于规则1更新后的晶格)
        if (get_bit(lattice_packed, i) == 0) { // 如果当前节点 i 的状态是0
            std::vector<int> neighbors = get_neighbors(i, p); // 获取其邻居
            for (int neighbor_idx : neighbors) { // 遍历所有邻居
                // 确保邻居索引有效 (尽管 get_neighbors 通常只返回有效索引)
                if (neighbor_idx >= 0 && neighbor_idx < p.total_nodes_for_bitpacking) {
                     to_be_zeroed[neighbor_idx] = true; // 标记该邻居节点将要变为0
                }
            }
        }
    }
    // 应用规则2的变更：将所有被标记的节点（即0状态节点的邻居）变为0状态
    for(long long i = 0; i < p.total_nodes_for_bitpacking; ++i) { // 遍历所有节点
        if(to_be_zeroed[i]) { // 如果节点 i 被标记为需要变为0
            set_bit(lattice_packed, i, 0); // 将其状态设置为0
        }
    }
} // apply_rules_packed 函数定义结束



class HopcroftKarp { // HopcroftKarp 类：封装了 Hopcroft-Karp 算法，用于在二分图中找到最大基数匹配 (Maximum Cardinality Matching)
    
    private:
    const int INF = std::numeric_limits<int>::max(); // 定义常量 INF (无穷大)，用于表示BFS过程中的距离，通常设为int类型的最大值
    std::vector<int> match_U, match_V, dist; // match_U: U中顶点到其匹配的V中顶点的映射; match_V: V中顶点到其匹配的U中顶点的映射; dist: BFS中U中顶点的距离标号
    int nU_hk, nV_hk; // 分别存储二分图U部和V部的顶点数量 (hk后缀用于区分命名空间内的变量)

    bool bfs(const std::vector<std::vector<int>>& current_adj_U_to_V) { // Hopcroft-Karp算法的BFS阶段：构建层次图，寻找最短增广路径
        // current_adj_U_to_V: 从U部顶点到V部顶点的邻接表
        std::fill(dist.begin(), dist.end(), INF); // 初始化所有U部顶点的距离为无穷大
        std::vector<int> q; // BFS 
        q.reserve(nU_hk); // 为队列预分配空间，提高效率
        for (int u = 0; u < nU_hk; ++u) { // 遍历U部的所有顶点
            if (match_U[u] == -1) { // 如果U部的顶点u是未匹配点
                q.push_back(u); // 将其加入队列作为BFS的起点
                dist[u] = 0; // 其距离标号设为0
            }
        }

        int head = 0; // 队列的头指针 (用于手动管理队列，避免频繁的pop_front)
        bool found_augmenting_path_to_unmatched_V = false; // 标记是否找到了通往V部未匹配点的增广路径的前半部分
        while(head < q.size()){ // 当队列非空时
            int u = q[head++]; // 取出队首顶点u
            for (int v_node_in_V : current_adj_U_to_V[u]) { // 遍历u在V部的所有邻居v
                if (match_V[v_node_in_V] == -1) { // 如果邻居v是V部的未匹配点
                    found_augmenting_path_to_unmatched_V = true; // 标记已找到至少一条可能的增广路径终点
                    // BFS会继续完成当前长度的所有最短路径层次的构建，而不是在此立即返回
                }
                // 如果v已匹配 (match_V[v_node_in_V] != -1)，并且v所匹配的U部顶点 (match_V[v_node_in_V]) 的距离尚未计算 (为INF)
                // 这意味着我们发现了一条交替路径 u -> v ~ u' (其中 ~ 表示匹配边)，可以扩展层次
                if (match_V[v_node_in_V] == -1 || (dist[match_V[v_node_in_V]] == INF)) { // 如果v未匹配，或v匹配的u'的距离是INF
                     if(match_V[v_node_in_V] != -1) { // 确保v是已匹配的 (从而 match_V[v_node_in_V] 是有效的U部顶点)
                        dist[match_V[v_node_in_V]] = dist[u] + 1; // 更新u'的距离标号 (比u的距离大1)
                        q.push_back(match_V[v_node_in_V]); // 将u'加入队列，继续BFS
                     }
                }
            }
        }
        return found_augmenting_path_to_unmatched_V; // 返回BFS是否找到了任何通往V部未匹配点的路径，这表明可能还存在增广路径
    } // bfs 函数定义结束

    bool dfs(int u, const std::vector<std::vector<int>>& current_adj_U_to_V) { // Hopcroft-Karp算法的DFS阶段：在BFS构建的层次图上寻找多条不相交的最短增广路径
        // u: 当前DFS访问的U部顶点
        // current_adj_U_to_V: U到V的邻接表
        for (int v_node_in_V : current_adj_U_to_V[u]) { // 遍历u在V部的所有邻居v
            // 条件1: v是未匹配点 (match_V[v_node_in_V] == -1)
            // 条件2: v已匹配，但从v的匹配点u' (match_V[v_node_in_V]) 出发能找到增广路径 (递归调用dfs)
            //         并且这条边 (u,v) 和 (v,u') 属于BFS构建的最短路径层次 (dist[u'] == dist[u] + 1)
            if (match_V[v_node_in_V] == -1 || (dist[match_V[v_node_in_V]] == dist[u] + 1 && dfs(match_V[v_node_in_V], current_adj_U_to_V))) {
                match_V[v_node_in_V] = u; // 找到增广路径，更新匹配：v与u匹配
                match_U[u] = v_node_in_V; // 更新匹配：u与v匹配
                return true; // 表示从u出发找到了一条增广路径
            }
        }
        dist[u] = INF; // 如果从u出发未能找到增广路径 (在此DFS调用中)，将其距离标号设为INF，表示此轮DFS不再访问它
        return false; // 表示从u出发未能找到增广路径
    } // dfs 函数定义结束

    public:

    const std::vector<int>& get_match_U() const { return match_U; }

    const std::vector<int>& get_match_V() const { return match_V; }

    const std::vector<int>& get_dist() const { return dist; }

    int run(int num_U, int num_V, const std::vector<std::vector<int>>& adj_list_U) { // Hopcroft-Karp算法的主执行函数
        // num_U: U部的顶点数量
        // num_V: V部的顶点数量
        // adj_list_U: U到V的邻接表
        nU_hk = num_U; // 设置U部顶点数
        nV_hk = num_V; // 设置V部顶点数

        match_U.assign(nU_hk, -1); // 初始化U部顶点的匹配信息，-1表示未匹配
        match_V.assign(nV_hk, -1); // 初始化V部顶点的匹配信息，-1表示未匹配
        if (nU_hk > 0) dist.resize(nU_hk); // 如果U部非空，调整dist向量大小以存储距离标号
        else dist.clear(); // 否则清空dist

        int counter = 0; // 用于计数BFS阶段的迭代次数 (可选，通常用于调试或性能分析)

        int matching_size = 0; // 初始化最大匹配的大小为0
        while (nU_hk > 0 && bfs(adj_list_U)) { // 当U部非空且BFS阶段能够找到增广路径的层次时，循环继续
            // (BFS返回true意味着可能存在更多增广路径)
            counter++; // 增加BFS迭代计数 (可选，通常用于调试或性能分析)
            for (int u = 0; u < nU_hk; ++u) { // 遍历U部的所有顶点
                if (match_U[u] == -1 && dfs(u, adj_list_U)) { // 如果顶点u是未匹配点，并且从u出发DFS能找到一条增广路径
                    matching_size++; // 最大匹配的大小增加1
                }
            }
        }
       //std::cout << "Hopcroft-Karp BFS iterations: " << counter << std::endl;  // 输出BFS迭代次数 (可选，通常用于调试或性能分析) 
        return matching_size; // 返回计算得到的最大匹配的大小
    } // run 函数定义结束
}; // HopcroftKarp 命名空间结束


std::vector<int> get_mis_nodes_indices( // 函数 get_mis_nodes_indices：根据二分图的最大匹配结果，利用Konig定理相关的顶点覆盖构造方法，找出最大独立集(MIS)的节点索引
    int current_nU, int current_nV, // 当前二分图U部和V部的顶点数量
    const std::vector<std::vector<int>>& adj_U_to_V_graph, // U到V的邻接表 (使用局部索引)
    const std::vector<int>& current_match_U, // U部顶点的匹配情况 (值为V中对应匹配顶点的局部索引, -1表示未匹配)
    const std::vector<int>& current_match_V, // V部顶点的匹配情况 (值为U中对应匹配顶点的局部索引, -1表示未匹配)
    const std::vector<int>& U_global_indices, // U部局部索引到全局节点索引的映射
    const std::vector<int>& V_global_indices  // V部局部索引到全局节点索引的映射
) {
    // 算法基于：在二分图中，|最大独立集| + |最小顶点覆盖| = |总顶点数|，且 |最大匹配| = |最小顶点覆盖| (Konig定理)。
    // 构造最小顶点覆盖(MVC)的一种方法是：
    // 1. 令Z为从U中所有未匹配点开始，沿交替路径可达的所有顶点的集合。
    //    (交替路径：非匹配边 U->V，匹配边 V->U，非匹配边 U->V ...)
    // 2. 则 MVC = (U \ Z_U) U Z_V，其中 Z_U = Z ∩ U, Z_V = Z ∩ V。
    // 3. 相应地，最大独立集 MIS = Z_U U (V \ Z_V)。

    std::vector<bool> Z_U_local_bool(current_nU, false); // 标记U部顶点是否属于集合Z (Z_U)
    std::vector<bool> Z_V_local_bool(current_nV, false); // 标记V部顶点是否属于集合Z (Z_V)
    std::vector<int> q_bfs; // 用于构建集合Z的BFS队列
    q_bfs.reserve(current_nU + current_nV); // 预分配空间

    std::vector<bool> visited_U_for_Z(current_nU, false); // 标记U部顶点是否已作为BFS起点或已处理
    std::vector<bool> visited_V_for_Z(current_nV, false); // 标记V部顶点是否已在BFS中访问过

    // 步骤1 & 2: 构建集合Z (从U中未匹配点开始沿交替路径遍历)
    for (int u_local = 0; u_local < current_nU; ++u_local) { // 遍历U部的所有顶点 (局部索引)
        if (current_match_U[u_local] == -1 && !visited_U_for_Z[u_local]) { // 如果u_local是未匹配点且尚未作为BFS起点
            q_bfs.clear(); // 清空队列
            q_bfs.push_back(u_local); // 将u_local加入队列作为起点
            visited_U_for_Z[u_local] = true; // 标记为已访问 (作为起点)
            Z_U_local_bool[u_local] = true; // u_local属于Z_U

            int head = 0; // 队列头指针
            while(head < q_bfs.size()){ // 当队列非空
                int curr_u_local = q_bfs[head++]; // 取出U部的当前顶点curr_u_local (curr_u_local ∈ Z_U)

                // 从 curr_u_local 出发，沿图中的边 (非匹配边) 到V部的邻居
                for (int neighbor_v_local : adj_U_to_V_graph[curr_u_local]) { // 遍历curr_u_local的所有邻居neighbor_v_local (在V部)
                    // (curr_u_local, neighbor_v_local) 是一条非匹配边，因为curr_u_local在BFS路径上是通过匹配边从V部过来的，或者它是起点(未匹配)
                    // 而交替路径要求从U到V走非匹配边。实际上这里是寻找所有从初始未匹配U点出发的“向前”路径。
                    // 如果是从Z中的U点出发，通过任意边（在此图中均为非匹配边，因为我们从U的匹配对象不再扩展）到达V点。
                    // 这里的逻辑是：如果u∈Z_U, v是u的邻居,则v∈Z_V。
                    if (!visited_V_for_Z[neighbor_v_local]) { // 如果V部的neighbor_v_local尚未被加入Z_V
                        visited_V_for_Z[neighbor_v_local] = true; // 标记为已访问
                        Z_V_local_bool[neighbor_v_local] = true; // neighbor_v_local属于Z_V

                        // 从 neighbor_v_local (∈ Z_V) 出发，如果它已匹配，则沿匹配边回到U部
                        if (current_match_V[neighbor_v_local] != -1 && !visited_U_for_Z[current_match_V[neighbor_v_local]]) {
                             int next_u_local = current_match_V[neighbor_v_local]; // 获取neighbor_v_local在U部的匹配对象
                             visited_U_for_Z[next_u_local] = true; // 标记为已访问
                             Z_U_local_bool[next_u_local] = true; // next_u_local属于Z_U
                             q_bfs.push_back(next_u_local); // 将next_u_local加入队列继续扩展
                        }
                    }
                }
            }
        }
    }

    // 步骤 3: 计算最大独立集 MIS = Z_U U (V \ Z_V)
    std::vector<int> mis_global_indices; // 存储MIS中节点的全局索引
    mis_global_indices.reserve(current_nU + current_nV); // 为MIS列表预分配大致空间
    
    // 添加 Z_U 中的节点到MIS
    for (int u_local = 0; u_local < current_nU; ++u_local) { // 遍历U部的所有顶点
        if (Z_U_local_bool[u_local]) { // 如果u_local属于Z_U
            mis_global_indices.push_back(U_global_indices[u_local]); // 将其对应的全局索引加入MIS列表
        }
    }
    // 添加 (V \ Z_V) 中的节点到MIS
    for (int v_local = 0; v_local < current_nV; ++v_local) { // 遍历V部的所有顶点
        if (!Z_V_local_bool[v_local]) { // 如果v_local不属于Z_V
            mis_global_indices.push_back(V_global_indices[v_local]); // 将其对应的全局索引加入MIS列表
        }
    }

    std::sort(mis_global_indices.begin(), mis_global_indices.end()); // 对MIS节点索引进行排序 (可选，但通常使输出规范)
    return mis_global_indices; // 返回最大独立集的全局节点索引列表
} // get_mis_nodes_indices 函数定义结束

std::vector<int> get_undetermined_nodes_indices(int current_nU, int current_nV, // 当前二分图U部和V部的顶点数量
    const std::vector<std::vector<int>>& adj_U_to_V_graph, // U到V的邻接表 (使用局部索引)
    const std::vector<int>& current_match_U, // U部顶点的匹配情况 (值为V中对应匹配顶点的局部索引, -1表示未匹配)
    const std::vector<int>& current_match_V, // V部顶点的匹配情况 (值为U中对应匹配顶点的局部索引, -1表示未匹配)
    const std::vector<int>& U_global_indices, // U部局部索引到全局节点索引的映射
    const std::vector<int>& V_global_indices  // V部局部索引到全局节点索引的映射
    )
{
    const int UNDETERMINED = 0;
    const int POSITIVE_BACKBONE = 1;
    const int NEGATIVE_BACKBONE = 2;

    // 状态数组
    std::vector<int> state_U(current_nU, UNDETERMINED);
    std::vector<int> state_V(current_nV, UNDETERMINED);

    // 构建V部到U部的邻接表（局部索引）
    std::vector<std::vector<int>> adj_V_to_U_graph(current_nV);
    for (int u = 0; u < current_nU; ++u) {
        for (int v : adj_U_to_V_graph[u]) {
            adj_V_to_U_graph[v].push_back(u);
        }
    }

    // 正主干队列
    std::vector<int> queue_U_positive;
    std::vector<int> queue_V_positive;
    queue_U_positive.reserve(current_nU);
    queue_V_positive.reserve(current_nV);

    // 1. 初始化：所有未匹配节点为正主干
    for (int u = 0; u < current_nU; ++u) {
        if (current_match_U[u] == -1) {
            state_U[u] = POSITIVE_BACKBONE;
            queue_U_positive.push_back(u);
        }
    }
    for (int v = 0; v < current_nV; ++v) {
        if (current_match_V[v] == -1) {
            state_V[v] = POSITIVE_BACKBONE;
            queue_V_positive.push_back(v);
        }
    }

    size_t head_U = 0, head_V = 0;

    // 2. 递归传播
    while (head_U < queue_U_positive.size() || head_V < queue_V_positive.size()) {
        // 处理U部正主干
        while (head_U < queue_U_positive.size()) {
            int u = queue_U_positive[head_U++];
            for (int v : adj_U_to_V_graph[u]) {
                if (state_V[v] == UNDETERMINED) {
                    state_V[v] = NEGATIVE_BACKBONE;
                    // 若v有匹配边，则匹配节点必须为正主干
                    if (current_match_V[v] != -1) {
                        int u_prime = current_match_V[v];
                        if (state_U[u_prime] == UNDETERMINED) {
                            state_U[u_prime] = POSITIVE_BACKBONE;
                            queue_U_positive.push_back(u_prime);
                        }
                    }
                }
            }
        }

        // 处理V部正主干
        while (head_V < queue_V_positive.size()) {
            int v = queue_V_positive[head_V++];
            for (int u : adj_V_to_U_graph[v]) {
                if (state_U[u] == UNDETERMINED) {
                    state_U[u] = NEGATIVE_BACKBONE;
                    // 若u有匹配边，则匹配节点必须为正主干
                    if (current_match_U[u] != -1) {
                        int v_prime = current_match_U[u];
                        if (state_V[v_prime] == UNDETERMINED) {
                            state_V[v_prime] = POSITIVE_BACKBONE;
                            queue_V_positive.push_back(v_prime);
                        }
                    }
                }
            }
        }
    }

    // 3. 收集所有未冻结节点（状态为UNDETERMINED）的全局索引
    std::vector<int> undetermined_global_indices;
    undetermined_global_indices.reserve(current_nU + current_nV);

    for (int u = 0; u < current_nU; ++u) {
        if (state_U[u] == UNDETERMINED) {
            undetermined_global_indices.push_back(U_global_indices[u]);
        }
    }
    for (int v = 0; v < current_nV; ++v) {
        if (state_V[v] == UNDETERMINED) {
            undetermined_global_indices.push_back(V_global_indices[v]);
        }
    }

    // 按全局索引排序（便于后续处理）
    std::sort(undetermined_global_indices.begin(), undetermined_global_indices.end());
    return undetermined_global_indices;
}
std::string format_rho_string(double rho_val) { // 函数 format_rho_string：将double类型的rho值格式化为字符串，去除末尾多余的零和小数点
    // rho_val: 需要格式化的double类型数值
    std::ostringstream oss; // 创建一个输出字符串流对象
    oss << rho_val; // 将double值写入字符串流
    std::string s = oss.str(); // 从字符串流中获取转换后的字符串
    size_t dot_pos = s.find('.'); // 查找字符串中小数点的位置
    if (dot_pos != std::string::npos) { // 如果找到了小数点 (即字符串包含小数部分)
        s.erase(s.find_last_not_of('0') + 1, std::string::npos); // 删除字符串末尾所有连续的'0'
        if (!s.empty() && s.back() == '.') { // 如果删除末尾'0'后，字符串的最后一个字符是小数点
            s.pop_back(); // 删除这个多余的小数点 (例如 "1.00" -> "1.", then "1." -> "1")
        }
    }
    return s; // 返回格式化后的字符串
} // format_rho_string 函数定义结束


struct MisResult { // 定义结构体 MisResult，用于封装最大独立集 (MIS) 计算过程中的各种相关结果数据
    std::vector<int> mis_nodes_global_indices; // 存储构成最大独立集的节点的全局索引列表
    std::vector<int> undetermined_node_indices;
    std::vector<uint64_t> mis_lattice_representation_packed; // MIS的位打包表示：一个与原始晶格同样大小的位向量，其中为1的位表示对应节点属于MIS
    std::vector<uint64_t> undetermined_node_packed;
    int nU; // 在为MIS构建的二分图中，U部分的顶点数量
    int nV; // 在为MIS构建的二分图中，V部分的顶点数量
    std::vector<int> match_U; // U部顶点的匹配对象 (V部局部索引)，-1表示未匹配
    std::vector<int> match_V; // V部顶点的匹配对象 (U部局部索引)，-1表示未匹配
    int mcm_size; // 计算得到的最大基数匹配 (Maximum Cardinality Matching) 的大小
    std::vector<int> U_local_to_global_idx; // U部分内部局部索引到原始晶格全局节点索引的映射
    std::vector<int> V_local_to_global_idx; // V部分内部局部索引到原始晶格全局节点索引的映射
    std::vector<std::vector<int>> final_adj_U_to_V_local; // 构建的二分图中，从U部顶点到V部顶点的邻接表 (使用局部索引)
}; // MisResult 结构体定义结束



MisResult calculate_mis_representation_packed( // 函数 calculate_mis_representation_packed：根据给定的晶格状态 (位打包)，计算其对应的最大独立集 (MIS) 及其位打包表示
    const std::vector<uint64_t>& lattice_packed, // 输入的最终晶格状态 (位打包形式)，其中值为1的节点构成待处理的图
    const Params& p // 程序参数，包含维度、尺寸等信息
) {
    MisResult result; // 创建一个MisResult对象来存储计算结果
    long long num_blocks = get_num_blocks(p.total_nodes_for_bitpacking); // 计算表示整个晶格的MIS状态所需的64位块数
    result.mis_lattice_representation_packed.assign(num_blocks, 0ULL); // 初始化MIS的位打包表示，所有位为0
    result.undetermined_node_packed.assign(num_blocks,0ull);
    // 创建映射表：全局节点索引 -> U/V部分的局部索引。-1表示节点不活跃(值为0)或不属于对应部分。
    std::vector<int> global_to_U_local_map(p.total_nodes_for_bitpacking, -1);
    std::vector<int> global_to_V_local_map(p.total_nodes_for_bitpacking, -1);
    
    // 预估活跃节点 (晶格中值为1的节点) 的数量，并为U/V部分的索引映射表预分配内存
    // (1.0 - p.rho) 是初始1的比例，这里作为一个粗略的活跃节点比例估计
    size_t active_nodes_estimate = p.total_nodes_for_bitpacking > 0 ? static_cast<size_t>(p.total_nodes_for_bitpacking * (1.0 - p.rho)) : 0; 
    result.U_local_to_global_idx.reserve(active_nodes_estimate / 2 + 10); // U部分预估大小 (大致为活跃节点一半)
    result.V_local_to_global_idx.reserve(active_nodes_estimate / 2 + 10); // V部分预估大小

    // 步骤1: 构建二分图的U和V顶点集
    // 遍历所有晶格节点，将状态为1的节点根据其坐标和的奇偶性分配到U或V部分
    for (long long i = 0; i < p.total_nodes_for_bitpacking; ++i) { // 遍历所有节点
        if (get_bit(lattice_packed, i) == 1) { // 如果节点 i 的状态为1 (即活跃节点，参与构建二分图)
            std::vector<int> coords = index_to_coords(i, p.L, p.W, p.d); // 获取节点 i 的 d 维坐标
            int coord_sum = 0; // 初始化坐标和
            for (int c : coords) coord_sum += c; // 计算所有维度坐标的总和

            if (coord_sum % 2 == 0) { // 如果坐标和为偶数
                global_to_U_local_map[i] = result.U_local_to_global_idx.size(); // 记录全局索引i到U部局部索引的映射
                result.U_local_to_global_idx.push_back(i); // 将节点i的全局索引加入U部的列表
            } else { // 如果坐标和为奇数
                global_to_V_local_map[i] = result.V_local_to_global_idx.size(); // 记录全局索引i到V部局部索引的映射
                result.V_local_to_global_idx.push_back(i); // 将节点i的全局索引加入V部的列表
            }
        }
    }

    result.nU = result.U_local_to_global_idx.size(); // U部分的顶点数量
    result.nV = result.V_local_to_global_idx.size(); // V部分的顶点数量
    result.final_adj_U_to_V_local.assign(result.nU, std::vector<int>()); // 初始化U到V的邻接表 (大小为nU，每个元素是V中局部索引的列表)

    // 步骤2: 构建二分图的边 (仅当U和V部分都非空时)
    if (result.nU > 0 && result.nV > 0) {
        for (int u_local_idx = 0; u_local_idx < result.nU; ++u_local_idx) { // 遍历U部分的每个顶点 (使用其局部索引)
            int u_global_idx = result.U_local_to_global_idx[u_local_idx]; // 获取该U部顶点的全局索引
            std::vector<int> u_coords = index_to_coords(u_global_idx, p.L, p.W, p.d); // 获取其 d 维坐标

            // 检查u_global_idx在各个维度上的邻居，如果邻居活跃且属于V部分，则添加一条边
            for (int dim_idx = 0; dim_idx < p.d; ++dim_idx) { // 遍历每个维度
                int dim_size = (dim_idx == 0) ? p.L : p.W; // 当前维度的大小
                if (dim_size <= 1) continue; // 如果维度大小为1，该维度无不同邻居

                long long multiplier = p.L_multipliers[dim_idx]; // 当前维度的乘数
                int original_coord_val = u_coords[dim_idx]; // u 在当前维度的坐标值
                int neighbor_global_idx; // 潜在邻居的全局索引
                
                // 检查正方向邻居
                bool p_is_valid = true; // 标记正方向邻居是否存在/有效
                if (p.periodic_boundary) { // 周期性边界
                    neighbor_global_idx = (original_coord_val == dim_size - 1) ? (u_global_idx - original_coord_val * multiplier) : (u_global_idx + multiplier);
                } else { // 非周期性边界
                    if (original_coord_val + 1 < dim_size) neighbor_global_idx = u_global_idx + multiplier;
                    else p_is_valid = false; // 超出边界，邻居无效
                }
                if (p_is_valid && neighbor_global_idx >= 0 && neighbor_global_idx < p.total_nodes_for_bitpacking) { // 如果邻居有效且在晶格范围内
                    int v_local_idx = global_to_V_local_map[neighbor_global_idx]; // 获取该邻居在V部分的局部索引
                    if (v_local_idx != -1) { // 如果邻居是活跃的V部节点 (v_local_idx != -1 表示它在V_local_to_global_idx中)
                        result.final_adj_U_to_V_local[u_local_idx].push_back(v_local_idx); // 添加从u_local_idx到v_local_idx的边
                    }
                }

                // 检查负方向邻居
                bool m_is_valid = true; // 标记负方向邻居是否存在/有效
                if (p.periodic_boundary) { // 周期性边界
                    neighbor_global_idx = (original_coord_val == 0) ? (u_global_idx + (dim_size - 1) * multiplier) : (u_global_idx - multiplier);
                } else { // 非周期性边界
                    if (original_coord_val > 0) neighbor_global_idx = u_global_idx - multiplier;
                    else m_is_valid = false; // 超出边界，邻居无效
                }
                if (m_is_valid && neighbor_global_idx >= 0 && neighbor_global_idx < p.total_nodes_for_bitpacking) { // 如果邻居有效且在晶格范围内
                    int v_local_idx = global_to_V_local_map[neighbor_global_idx]; // 获取该邻居在V部分的局部索引
                     if (v_local_idx != -1) { // 如果邻居是活跃的V部节点
                       result.final_adj_U_to_V_local[u_local_idx].push_back(v_local_idx); // 添加边
                    }
                }
            }
            
            // 对每个U部顶点的邻接列表进行排序和去重 (以处理因L=2或W=2等特殊情况可能产生的重复边)
            if ((p.L == 2 || p.W == 2) && p.d > 1 && !result.final_adj_U_to_V_local[u_local_idx].empty()) {
                std::sort(result.final_adj_U_to_V_local[u_local_idx].begin(), result.final_adj_U_to_V_local[u_local_idx].end());
                result.final_adj_U_to_V_local[u_local_idx].erase(
                    std::unique(result.final_adj_U_to_V_local[u_local_idx].begin(), result.final_adj_U_to_V_local[u_local_idx].end()),
                    result.final_adj_U_to_V_local[u_local_idx].end()
                );
            }
        }
    }

    // 步骤3: 计算最大匹配 (MCM) 和最大独立集 (MIS)
    result.mcm_size = 0; // 初始化最大匹配的大小
    if (result.nU > 0 && result.nV > 0) { // 如果U和V部分都非空，可以运行Hopcroft-Karp算法
        HopcroftKarp solver;  // 创建求解器实例
        result.mcm_size = solver.run(result.nU, result.nV, result.final_adj_U_to_V_local); // 执行Hopcroft-Karp算法，得到最大匹配数
        result.match_U = solver.get_match_U();
        result.match_V = solver.get_match_V();
        // 根据最大匹配结果，使用Konig定理相关方法找到MIS的节点，并使用Freeze-influence算法找出所有type-*格点
        result.mis_nodes_global_indices = get_mis_nodes_indices(
            result.nU, result.nV, // U和V部分的顶点数
            result.final_adj_U_to_V_local, // U到V的邻接表
            solver.get_match_U(),         // Hopcroft-Karp算法得到的U部匹配信息
            solver.get_match_V(),         // Hopcroft-Karp算法得到的V部匹配信息
            result.U_local_to_global_idx,  // U部局部索引到全局索引的映射
            result.V_local_to_global_idx   // V部局部索引到全局索引的映射
        );
        result.undetermined_node_indices = get_undetermined_nodes_indices(result.nU, result.nV, // U和V部分的顶点数
            result.final_adj_U_to_V_local, // U到V的邻接表
            solver.get_match_U(),         // Hopcroft-Karp算法得到的U部匹配信息
            solver.get_match_V(),         // Hopcroft-Karp算法得到的V部匹配信息
            result.U_local_to_global_idx,  // U部局部索引到全局索引的映射
            result.V_local_to_global_idx   // V部局部索引到全局索引的映射
            );
    } else { // 如果U或V部分为空 (或两者皆空)，则二分图是退化的
        if (result.nU > 0) { // 如果只有U部分非空 (nV=0)，则U中所有节点构成MIS
            result.mis_nodes_global_indices = result.U_local_to_global_idx;
        } else { // 如果只有V部分非空 (nU=0)，或两者皆空，则V中所有节点构成MIS (若V也空，则MIS为空)
            result.mis_nodes_global_indices = result.V_local_to_global_idx;
        }
        result.mcm_size = 0; // 此时最大匹配数为0
        result.match_U.clear(); result.match_V.clear();
    }

    // 步骤4: 生成MIS的位打包表示
    for (int mis_node_idx : result.mis_nodes_global_indices) { // 遍历所有属于MIS的节点的全局索引
        if (mis_node_idx >= 0 && mis_node_idx < p.total_nodes_for_bitpacking) { // 确保索引有效
            set_bit(result.mis_lattice_representation_packed, mis_node_idx, 1); // 在MIS的位打包表示中，将对应节点的位置1
        }
    }
    for(int undetermined_node_idx:result.undetermined_node_indices)
    {
        if(undetermined_node_idx >= 0 && undetermined_node_idx < p.total_nodes_for_bitpacking)
        {
            set_bit(result.undetermined_node_packed,undetermined_node_idx,1);
        }
    }
    
    return result; // 返回包含所有MIS相关计算结果的MisResult对象
} // calculate_mis_representation_packed 函数定义结束



std::vector<uint64_t> process_boundary_nodes_optimized(const std::vector<uint64_t>& lattice_packed, // 函数 process_boundary_nodes_optimized：优化地处理边界节点。根据类型 'A' 或 'B'，将特定边界节点（基于其坐标和的奇偶性）的状态置为0。
                                                      const Params& p, // 程序参数，包含维度、尺寸等信息
                                                      char type) { // 处理类型：'A'表示坐标和为偶数的边界点置0, 'B'表示坐标和为奇数的边界点置0
    
    std::vector<uint64_t> result = lattice_packed; // 创建输入晶格的副本，修改将在此副本上进行
    
    // 基本参数检查：如果维度无效或总节点数为0，则不进行处理
    if (p.d <= 0 || p.total_nodes_for_bitpacking <= 0) {
        return result; // 直接返回原始晶格副本
    }
    
    // 类型参数检查：确保类型是 'A' 或 'B'
    if (type != 'A' && type != 'B') {
        std::cerr << "Error: type must be 'A' or 'B'" << std::endl; // 输出错误信息
        return result; // 返回原始晶格副本 (或者可以考虑抛出异常)
    }
    
    // 存储每个维度的大小
    std::vector<int> dim_sizes(p.d); // 创建向量存储各维度的大小
    dim_sizes[0] = p.L; // 第一个维度 (索引0) 的大小为 L
    for (int i = 1; i < p.d; ++i) { // 其他维度 (索引1到d-1)
        dim_sizes[i] = p.W; // 大小为 W
    }
    
    // 预计算每个维度上哪些坐标值属于边界
    std::vector<std::vector<bool>> is_boundary_coord(p.d); // is_boundary_coord[dim_idx][coord_value] 为true表示是边界
    for (int dim = 0; dim < p.d; ++dim) { // 遍历每个维度
        is_boundary_coord[dim].resize(dim_sizes[dim], false); // 初始化当前维度的所有坐标为非边界
        if (dim_sizes[dim] > 0) { // 如果维度大小有效
            is_boundary_coord[dim][0] = true;  // 坐标0总是边界 (下界)
            if (dim_sizes[dim] > 1) { // 如果维度大小大于1，则存在不同的上界
                is_boundary_coord[dim][dim_sizes[dim] - 1] = true;  // 坐标 dim_size-1 也是边界 (上界)
            }
        }
    }
    
    // 获取位打包所需的总块数
    long long num_blocks = get_num_blocks(p.total_nodes_for_bitpacking);
    
    for (long long block_idx = 0; block_idx < num_blocks; ++block_idx) { // 遍历每个64位数据块
        uint64_t block_mask = 0ULL;  // 初始化当前块的掩码为0。掩码中为1的位表示对应节点需要被置为0。
        long long start_node = block_idx * 64; // 当前块处理的起始节点的一维索引
        long long end_node = std::min(start_node + 64, p.total_nodes_for_bitpacking); // 当前块处理的结束节点索引 (不包含end_node)
        
        // 遍历当前块中的每个节点 (对应一个位)
        for (long long node_idx = start_node; node_idx < end_node; ++node_idx) {
            
            bool is_boundary = false; // 标记当前节点是否位于晶格的任何边界上
            int coord_sum = 0; // 当前节点的各维度坐标值之和
            
            // 从一维索引 node_idx 计算多维坐标，并检查是否为边界点及计算坐标和
            long long temp_idx = node_idx; // 临时索引，用于从一维索引计算多维坐标
            
            for (int dim = p.d - 1; dim >= 0; --dim) { // 从最高维度(d-1)向最低维度(0)迭代，以分解索引
                int coord; // 当前维度的坐标值
                if (dim == 0) { // 当计算第0维 (L相关的维度) 的坐标时
                    coord = temp_idx % dim_sizes[0]; // temp_idx此时是原始索引除以所有W维度大小后的值，再对L取模得到第0维坐标
                } else { // 当计算其他维度 (W相关的维度, dim > 0) 的坐标时
                    coord = temp_idx % dim_sizes[dim]; // 对当前维度的大小W取模得到该维度坐标
                    temp_idx /= dim_sizes[dim]; // 更新temp_idx，除去当前维度的影响，为计算下一较低维度做准备
                }
                
                coord_sum += coord; // 累加坐标值
                
                // 检查当前坐标是否是该维度上的边界 (确保维度有效且坐标在范围内，再访问is_boundary_coord)
                if (dim_sizes[dim] > 0 && coord >= 0 && coord < dim_sizes[dim] && is_boundary_coord[dim][coord]) {
                    is_boundary = true; // 如果任何一个维度的坐标是边界，则该节点是边界节点
                                        // (注意：此处不break，因为仍需计算完整的coord_sum)
                }
            }
            
            // 如果当前节点是边界节点，则根据类型和坐标和的奇偶性决定是否将其状态置为0
            if (is_boundary) {
                // type 'A': 若坐标和为偶数，则该边界点需要置0
                // type 'B': 若坐标和为奇数，则该边界点需要置0
                bool should_clear = (type == 'A') ? ((coord_sum % 2) == 0) : ((coord_sum % 2) == 1); // 判断是否满足置0条件
                if (should_clear) { // 如果需要置0
                    int bit_pos = node_idx - start_node; // 计算该节点在当前64位块中的位偏移 (0-63)
                    block_mask |= (1ULL << bit_pos); // 在掩码的对应位置1
                }
            }
        }
        
        // 应用掩码：将当前块中所有被标记 (掩码对应位为1) 的节点状态置为0
        result[block_idx] &= ~block_mask; // 使用按位与和取反的掩码 (~) 来清零指定位，其余位保持不变
    }
    
    return result; // 返回处理后的晶格状态 (位打包形式)
} // process_boundary_nodes_optimized 函数定义结束


std::vector<uint64_t> process_boundary_nodes_ultra_fast(const std::vector<uint64_t>& lattice_packed, // 函数 process_boundary_nodes_ultra_fast：极快处理边界节点，针对低维度 (d=1, d=2) 有特化实现，其他情况回退到 optimized 版本
                                                        const Params& p) { // 处理类型 'A' 或 'B'
    std::vector<uint64_t> result = lattice_packed; // 创建输入晶格的副本
    char type = p.boundary_prefer;
    // 基本参数检查
    if (p.d <= 0 || p.total_nodes_for_bitpacking <= 0) {
        return result; // 无效参数则直接返回
    }
    
    // 类型参数检查
    if (type != 'A' && type != 'B') {
        std::cerr << "Error: type must be 'A' or 'B'" << std::endl;
        return result; // 无效类型则直接返回
    }
    
    // 特例处理：d=1 (一维链)
    if (p.d == 1) {
        // 一维链的边界点是索引 0 和 L-1
        if (p.L > 0) { // 处理第一个边界点 (索引0)，前提是链长有效
            int coord_sum_0 = 0; // 坐标是(0)，坐标和为0
            // 根据类型和坐标和奇偶性判断是否置0
            if ((type == 'A' && coord_sum_0 % 2 == 0) || (type == 'B' && coord_sum_0 % 2 == 1)) {
                set_bit(result, 0, 0); // 将索引0的位设置为0
            }
        }
        if (p.L > 1) { // 处理第二个边界点 (索引 L-1)，前提是链长大于1 (从而有两个不同边界点)
            long long last_idx = p.L - 1; // 最后一个点的索引
            int coord_sum_last = p.L - 1; // 坐标是(L-1)，坐标和为 L-1
            // 根据类型和坐标和奇偶性判断是否置0
            if ((type == 'A' && coord_sum_last % 2 == 0) || (type == 'B' && coord_sum_last % 2 == 1)) {
                set_bit(result, last_idx, 0); // 将索引 last_idx 的位设置为0
            }
        }
        return result; // d=1 情况处理完毕，返回结果
    }
    
    // 特例处理：d=2 (二维 L x W 网格)
    if (p.d == 2) {
        // 遍历四条边上的节点并处理
        // 边1: y=0 (第一行，所有列x从0到L-1)
        for (int x = 0; x < p.L; ++x) {
            long long idx = coords_to_index({x,0}, p.L, p.W, p.d) ;  // 节点(x,0)的一维索引
            int sum = x + 0; // 坐标和
            if ((type == 'A' && sum % 2 == 0) || (type == 'B' && sum % 2 == 1)) { // 判断是否置0
                set_bit(result, idx, 0);
            }
        }
        
        // 边2: y=W-1 (最后一行)，仅当 W > 1 时这条边与第一条边不同
        if (p.W > 1) {
            for (int x = 0; x < p.L; ++x) {
                long long idx = coords_to_index({x, p.W-1}, p.L, p.W, p.d); // 节点(x,W-1)的一维索引
                int sum = x + (p.W - 1); // 坐标和
                if ((type == 'A' && sum % 2 == 0) || (type == 'B' && sum % 2 == 1)) { // 判断是否置0
                    set_bit(result, idx, 0);
                }
            }
        }
        
        // 边3: x=0 (第一列，不含已处理的角点 y=0 和 y=W-1)
        // y 从 1 遍历到 W-2
        for (int y = 1; y < p.W - 1; ++y) { 
            long long idx = coords_to_index({0,y}, p.L, p.W, p.d); // 节点(0,y)的一维索引
            int sum = 0 + y; // 坐标和
            if ((type == 'A' && sum % 2 == 0) || (type == 'B' && sum % 2 == 1)) { // 判断是否置0
                set_bit(result, idx, 0);
            }
        }
        
        // 边4: x=L-1 (最后一列，不含已处理的角点 y=0 和 y=W-1)，仅当 L > 1
        if (p.L > 1) {
            for (int y = 1; y < p.W - 1; ++y) {
                long long idx = coords_to_index({p.L-1, y}, p.L, p.W, p.d); // 节点(L-1,y)的一维索引
                int sum = (p.L - 1) + y; // 坐标和
                if ((type == 'A' && sum % 2 == 0) || (type == 'B' && sum % 2 == 1)) { // 判断是否置0
                    set_bit(result, idx, 0);
                }
            }
        }
        return result; // d=2 情况处理完毕，返回结果
    }
    
    // 对于更高维度 (d > 2) 或其他未被特例覆盖的情况，回退到通用的 optimized 版本进行处理
    // 注意：这里传递的是原始的 lattice_packed，而不是已经可能被修改的 result。
    // 这意味着如果 d=1 或 d=2 的特例不满足，则完全由 optimized 版本处理原始数据。
    return process_boundary_nodes_optimized(lattice_packed, p, type); // 调用优化版本处理
} // process_boundary_nodes_ultra_fast 函数定义结束


std::vector<uint64_t> get_largest_connected_component_packed(const std::vector<uint64_t>& lattice_packed, const Params& p) {
    // 创建结果向量，初始化为全0
    long long num_blocks = get_num_blocks(p.total_nodes_for_bitpacking);
    std::vector<uint64_t> result_packed(num_blocks, 0ULL);
    
    if (p.total_nodes_for_bitpacking == 0) {
        return result_packed;
    }
    
    // 使用 BFS 遍历所有连通分量
    std::vector<bool> visited(p.total_nodes_for_bitpacking, false);
    std::vector<long long> largest_component;
    long long max_component_size = 0;
    
    for (long long i = 0; i < p.total_nodes_for_bitpacking; ++i) {
        // 如果节点活跃且未被访问，开始新的连通分量搜索
        if (get_bit(lattice_packed, i) == 1 && !visited[i]) {
            std::vector<long long> current_component;
            std::queue<long long> q;
            
            q.push(i);
            visited[i] = true;
            current_component.push_back(i);
            
            while (!q.empty()) {
                long long current_node = q.front();
                q.pop();
                
                // 获取当前节点的所有邻居
                std::vector<int> neighbors = get_neighbors(current_node, p);
                for (int neighbor : neighbors) {
                    if (neighbor >= 0 && neighbor < p.total_nodes_for_bitpacking && 
                        get_bit(lattice_packed, neighbor) == 1 && !visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                        current_component.push_back(neighbor);
                    }
                }
            }
            
            // 更新最大连通分量
            if (current_component.size() > max_component_size) {
                max_component_size = current_component.size();
                largest_component = std::move(current_component);
            }
        }
    }
    
    // 在结果向量中设置最大连通分量的节点为活跃
    for (long long node_index : largest_component) {
        set_bit(result_packed, node_index, 1);
    }
    
    return result_packed;
}



std::vector<uint32_t> get_random_seed_list(const Params& p)
{
    std::vector<uint32_t> seed_list; seed_list.reserve(p.num_samples);
    if(p.seed == 0)
    {
        std::random_device rd;
        for(int i =0; i < p.num_samples; i++)
        {
            uint32_t seed = rd();
            seed_list.push_back(seed);
        }
    }
    else
    {
        std::mt19937 gen(p.seed);
        for(int i =0; i < p.num_samples; i++)
        {
            uint32_t seed = gen();
            seed_list.push_back(seed);
        }
    }
    return seed_list;
}


template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_.notify_one();
    }
    
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

// 样本任务结构
struct SampleTask {
    unsigned int seed;
    int sample_id;
};


void save_unfrozen_reduced_graph(const MisResult& mis_result, const Params& p, const std::string& output_dir) {
    std::string filename = output_dir + "/unfrozen_reduced_graph.txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 构建全局索引到局部信息的映射
    std::map<int, int> global_to_part; // 0:U, 1:V
    std::map<int, int> global_to_local;
    for (int u = 0; u < mis_result.nU; ++u) {
        int g = mis_result.U_local_to_global_idx[u];
        global_to_part[g] = 0;
        global_to_local[g] = u;
    }
    for (int v = 0; v < mis_result.nV; ++v) {
        int g = mis_result.V_local_to_global_idx[v];
        global_to_part[g] = 1;
        global_to_local[g] = v;
    }
    outfile << "Unfrozen reduced graph size: " << mis_result.undetermined_node_indices.size() << std::endl;
    // 未冻结节点集合
    std::set<int> unfrozen_set(mis_result.undetermined_node_indices.begin(), mis_result.undetermined_node_indices.end());

    // 构建V到U的邻接表
    std::vector<std::vector<int>> V_to_U_adj(mis_result.nV);
    for (int u = 0; u < mis_result.nU; ++u) {
        for (int v : mis_result.final_adj_U_to_V_local[u]) {
            V_to_U_adj[v].push_back(u);
        }
    }

    // 为每个未冻结节点输出其邻居信息
    for (int global_node : mis_result.undetermined_node_indices) {
        // 节点坐标字符串
        std::vector<int> coords = index_to_coords(global_node, p.L, p.W, p.d);
        std::ostringstream node_ss;
        node_ss << "(";
        for (size_t i = 0; i < coords.size(); ++i) {
            if (i > 0) node_ss << ",";
            node_ss << coords[i];
        }
        node_ss << ")";
        std::string node_coord = node_ss.str();

        int part = global_to_part[global_node];
        int local = global_to_local[global_node];
        std::vector<std::string> neighbor_entries;

        if (part == 0) { // U部节点
            for (int v_local : mis_result.final_adj_U_to_V_local[local]) {
                int neigh_global = mis_result.V_local_to_global_idx[v_local];
                if (unfrozen_set.find(neigh_global) == unfrozen_set.end()) continue;
                // 邻居坐标
                std::vector<int> ncoords = index_to_coords(neigh_global, p.L, p.W, p.d);
                std::ostringstream neigh_ss;
                neigh_ss << "(";
                for (size_t i = 0; i < ncoords.size(); ++i) {
                    if (i > 0) neigh_ss << ",";
                    neigh_ss << ncoords[i];
                }
                neigh_ss << ")";
                bool is_match = (mis_result.match_U[local] != -1 && mis_result.match_U[local] == v_local);
                neighbor_entries.push_back(neigh_ss.str() + (is_match ? " 1" : " 0"));
            }
        } else { // V部节点
            for (int u_local : V_to_U_adj[local]) {
                int neigh_global = mis_result.U_local_to_global_idx[u_local];
                if (unfrozen_set.find(neigh_global) == unfrozen_set.end()) continue;
                // 邻居坐标
                std::vector<int> ncoords = index_to_coords(neigh_global, p.L, p.W, p.d);
                std::ostringstream neigh_ss;
                neigh_ss << "(";
                for (size_t i = 0; i < ncoords.size(); ++i) {
                    if (i > 0) neigh_ss << ",";
                    neigh_ss << ncoords[i];
                }
                neigh_ss << ")";
                bool is_match = (mis_result.match_U[u_local] != -1 && mis_result.match_U[u_local] == local);
                neighbor_entries.push_back(neigh_ss.str() + (is_match ? " 1" : " 0"));
            }
        }

        // 输出节点行
        outfile << node_coord << ":";
        for (size_t i = 0; i < neighbor_entries.size(); ++i) {
            if (i > 0) outfile << ";";
            outfile << " " << neighbor_entries[i];
        }
        outfile << ";" << std::endl;
    }
    outfile.close();
}

void save_type_data_to_file(const std::vector<uint64_t>& lattice_packed,
                            const std::vector<uint64_t>& Largest_connected_component_packed,
                            const MisResult& mis_result,
                            const Params& p, int sample_id,
                            unsigned int sample_seed,
                            const std::string& output_filename,
                            const std::string& output_filename_lcc,
                            std::mutex& file1_mutex, std::mutex& file2_mutex)
{
    // 初始化计数器
    int N_active = 0; int N_lcc = 0;
    int N_A_1 = 0; int N_A_1_lcc = 0; int N_B_1 = 0; int N_B_1_lcc = 0;
    int N_A_2 = 0; int N_A_2_lcc = 0; int N_B_2 = 0; int N_B_2_lcc = 0;
    int N_A_star = 0; int N_A_star_lcc = 0; int N_B_star = 0; int N_B_star_lcc = 0;

    // 预计算快速判断节点是否属于 MIS、未确定、LCC 的位打包数组
    const auto& mis_packed = mis_result.mis_lattice_representation_packed;
    const auto& undet_packed = mis_result.undetermined_node_packed;
    const auto& lcc_packed = Largest_connected_component_packed;

    for (long long idx = 0; idx < p.total_nodes_for_bitpacking; ++idx) {
        // 只处理活跃节点（晶格状态为1）
        if (get_bit(lattice_packed, idx) == 0) continue;

        // 获取坐标和奇偶性（A类=偶，B类=奇）
        std::vector<int> coords = index_to_coords(idx, p.L, p.W, p.d);
        int coord_sum = 0;
        for (int c : coords) coord_sum += c;
        bool is_A = (coord_sum % 2 == 0);  // A类（U部）

        // 判断节点类型
        bool in_mis = (get_bit(mis_packed, idx) == 1);
        bool in_undet = (get_bit(undet_packed, idx) == 1);
        bool in_lcc = (get_bit(lcc_packed, idx) == 1);

        // 更新整体统计
        N_active++;
        if(in_undet)
        {
            if(is_A){N_A_star++;}else{N_B_star++;}
        }
        else
        {
            if(in_mis)
            {
                if(is_A){N_A_1++;}else{N_B_1++;}
            }
            else
            {
                if(is_A){N_A_2++;}else{N_B_2++;}
            }
        }

        // 更新 LCC 统计
        if (in_lcc) {
            N_lcc++;
            if (in_undet) {
                if (is_A) N_A_star_lcc++; else N_B_star_lcc++;
            }
            else
            {
                if (in_mis) {
                if (is_A) N_A_1_lcc++; else N_B_1_lcc++;
            }
             else {
                if (is_A) N_A_2_lcc++; else N_B_2_lcc++;
            }
            }

            
        }
    }

    // 写入整体文件（线程安全）
    {
        std::lock_guard<std::mutex> lock(file1_mutex);
        std::ofstream out(output_filename, std::ios::app);
        if (out.is_open()) {
            out << sample_id << ", "
                << N_active << ", "
                << N_A_1 << ", " << N_B_1 << ", "
                << N_A_2 << ", " << N_B_2 << ", "
                << N_A_star << ", " << N_B_star << ", "
                << sample_seed << "\n";
        } else {
            std::cerr << "Error: Cannot open " << output_filename << " for writing.\n";
        }
    }

    // 写入 LCC 文件（线程安全）
    {
        std::lock_guard<std::mutex> lock(file2_mutex);
        std::ofstream out(output_filename_lcc, std::ios::app);
        if (out.is_open()) {
            out << sample_id << ", "
                << N_lcc << ", "
                << N_A_1_lcc << ", " << N_B_1_lcc << ", "
                << N_A_2_lcc << ", " << N_B_2_lcc << ", "
                << N_A_star_lcc << ", " << N_B_star_lcc << ", "
                << sample_seed << "\n";
        } else {
            std::cerr << "Error: Cannot open " << output_filename_lcc << " for writing.\n";
        }
    }
}

void run_sample(const Params& base_params, unsigned int sample_seed, int sample_id, 
                const std::string& filename, const std::string& lcc_filename,
                std::mutex& file1_mutex, std::mutex& file2_mutex )
{
    Params p = base_params;
    p.seed = sample_seed;
    std::cout << "Processing sample " << sample_id << " with random seed: " << p.seed <<std::endl;

    std::mt19937 gen(p.seed);//创建样本随机数引擎
    auto time_start_generation = std::chrono::high_resolution_clock::now(); // 记录晶格生成及规则应用开始的时间点
    std::vector<uint64_t> lattice_packed = generate_initial_lattice_packed(p.total_nodes_for_bitpacking, p.rho, gen); // 生成初始的位打包晶格状态
 
    apply_rules_packed(lattice_packed, p); // 对生成的晶格应用细胞自动机规则
    
    if(!p.periodic_boundary)
    {
        //在不采用周期性边界条件时，根据采用的外部占据环境对晶格边界进行处理
        lattice_packed = process_boundary_nodes_ultra_fast( lattice_packed, p); 
    }
    auto time_end_generation = std::chrono::high_resolution_clock::now(); // 记录晶格生成及规则应用结束的时间点
  
    std::chrono::duration<double, std::milli> generation_duration = time_end_generation - time_start_generation; // 计算此阶段的总耗时

    std::vector<uint64_t> Max_size_connected_component_packed = get_largest_connected_component_packed(lattice_packed,p);
   
    auto time_start_mis = std::chrono::high_resolution_clock::now(); // 记录MIS计算开始的时间点
    MisResult mis_result = calculate_mis_representation_packed(lattice_packed, p); // 计算处理后晶格的最大独立集 (MIS)
    auto time_end_mis = std::chrono::high_resolution_clock::now(); // 记录MIS计算结束的时间点
    std::chrono::duration<double, std::milli> mis_duration = time_end_mis - time_start_mis; // 计算MIS计算阶段的总耗时

    save_type_data_to_file(lattice_packed, Max_size_connected_component_packed, mis_result,
                       p, sample_id, sample_seed, filename, lcc_filename,
                       file1_mutex, file2_mutex);
}

// 工作线程函数
void worker_thread(ThreadSafeQueue<SampleTask>& task_queue, const Params& base_params,
                   const std::string& filename, const std::string& lcc_filename,
                   std::mutex& file1_mutex, std::mutex& file2_mutex, 
                   std::atomic<int>& completed_count, int total_samples) {
    SampleTask task;
    while (task_queue.try_pop(task)) {
        run_sample(base_params, task.seed, task.sample_id,
                   filename, lcc_filename,
                   file1_mutex, file2_mutex);
        int completed = ++completed_count;
        std::cout << "\rProgress: " << completed << "/" << total_samples << " samples completed \n";
    }
}


//g++ -O3 -march=native -pthread TemperatureKCM/MIS_RSG.cpp -o exe/MIS_RSG.exe
//MIS_parallel.exe 2 40 40 0.1 0.2 0.01 114514 1 100 B 10 
int main(int argc, char* argv[]) {
    Params p;
    // 默认值
    p.d = 2;
    p.L = 20;
    p.W = p.L;
    p.rho = 0.1;                 
    p.seed = 114514;
    p.periodic_boundary = true;
    p.num_samples = 100;
    p.boundary_prefer = 'B';
    unsigned int Max_thread_number = 5;
    std::string dir = "E:/CPPcode/cpp_source/MIS_type";

    if (argc > 1) p.d = std::stoi(argv[1]);
    if (argc > 2) p.L = std::stoi(argv[2]);
    if (argc > 3) p.W = std::stoi(argv[3]);

    double rho_start = 0.0, rho_end = 0.0, rho_step = 0.0;
    if (argc > 4) rho_start = std::stod(argv[4]);
    if (argc > 5) rho_end   = std::stod(argv[5]);
    if (argc > 6) rho_step  = std::stod(argv[6]);


    if (argc > 7) {
        try {
            long long temp_seed = std::stoll(argv[7]);
            if (temp_seed == 0) {
                p.seed = 0;
            } else if (temp_seed < 0 || temp_seed > std::numeric_limits<unsigned int>::max()) {
                std::cerr << "Warning: Provided seed " << temp_seed << " is out of range. Using random_device." << std::endl;
                p.seed = 0;
            } else {
                p.seed = static_cast<unsigned int>(temp_seed);
            }
        } catch (...) {
            std::cerr << "Warning: Invalid seed value. Using random_device." << std::endl;
            p.seed = 0;
        }
    }
    if (argc > 8) p.periodic_boundary = (std::stoi(argv[8]) == 1);
    if (argc > 9) p.num_samples = std::stoi(argv[9]);
    if (argc > 10) {
        std::string input_param = argv[10];
        std::transform(input_param.begin(), input_param.end(), input_param.begin(),
                       [](unsigned char c){ return std::toupper(c); });
        if (input_param == "A" || input_param == "B") {
            p.boundary_prefer = input_param[0];
        } else {
            std::cerr << "Warning: Invalid boundary preference. Using default " << p.boundary_prefer << std::endl;
        }
    }
    if (argc > 11) {
        int number_thread = std::stoi(argv[11]);
        if (number_thread > 0) {
            unsigned int num_thread = number_thread;
            Max_thread_number = std::min(num_thread, static_cast<unsigned int>(16));
        } else {
            std::cerr << "Invalid thread number, using default " << Max_thread_number << std::endl;
        }
    }
    if (argc > 12) {
        std::string input_dir = argv[12];
        bool Is_dir_exist = false;
#if __has_include(<filesystem>)
        if (fs::exists(input_dir)) Is_dir_exist = true;
#else
        struct stat st;
        if (stat(input_dir.c_str(), &st) != -1) Is_dir_exist = true;
#endif
        if (Is_dir_exist) {
            dir = input_dir;
        } else {
            std::cerr << "Can not find the input directory " << input_dir
                      << ", using default main output directory " << dir << std::endl;
        }
    }
    long long total_nodes_val = 1;
    try {
        if (p.d < 0 || p.L < 0 || p.W < 0) throw std::runtime_error("d, L and W must be non-negative");
        if (p.d == 0) {
            total_nodes_val = 1;
        } else if (p.d == 1) {
            total_nodes_val = p.L;
        } else {
            total_nodes_val = p.L;
            for (int i = 1; i < p.d; ++i) {
                unsigned __int128 temp = static_cast<unsigned __int128>(total_nodes_val) * p.W;
                if (temp > std::numeric_limits<long long>::max())
                    throw std::overflow_error("L × W^(d-1) too large for long long");
                total_nodes_val = static_cast<long long>(temp);
            }
        }
        if (total_nodes_val < 0) total_nodes_val = 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    p.total_nodes_for_bitpacking = total_nodes_val;


    p.L_multipliers.assign(p.d, 0LL);
    if (p.d > 0) {
        if (p.d == 1) {
            if (p.L > 0) p.L_multipliers[0] = 1LL;
        } else {
            if (p.W > 0) p.L_multipliers[p.d - 1] = 1LL;
            for (int k = p.d - 2; k >= 1; --k) {
                if (p.W == 0) {
                    p.L_multipliers[k] = 0LL;
                    continue;
                }
                unsigned __int128 temp = static_cast<unsigned __int128>(p.L_multipliers[k + 1]) * p.W;
                if (temp > std::numeric_limits<long long>::max()) {
                    std::cerr << "Warning: L_multiplier overflow." << std::endl;
                    p.L_multipliers[k] = std::numeric_limits<long long>::max();
                } else {
                    p.L_multipliers[k] = static_cast<long long>(temp);
                }
            }
            if (p.d >= 2 && p.W > 0 && p.L_multipliers[1] > 0) {
                unsigned __int128 temp = static_cast<unsigned __int128>(p.L_multipliers[1]) * p.W;
                if (temp > std::numeric_limits<long long>::max()) {
                    std::cerr << "Warning: L_multiplier overflow for first dimension." << std::endl;
                    p.L_multipliers[0] = std::numeric_limits<long long>::max();
                } else {
                    p.L_multipliers[0] = static_cast<long long>(temp);
                }
            } else if (p.d >= 2) {
                p.L_multipliers[0] = 0LL;
            }
        }
    }


    if (p.total_nodes_for_bitpacking == 0) {
        std::cout << "Total nodes is 0. Exiting." << std::endl;
        return 0;
    }

    std::cout << "Parameters: d=" << p.d << ", L=" << p.L << ", W=" << p.W
              << ", rho from " << rho_start << " to " << rho_end << " step " << rho_step
              << ", seed=" << p.seed << ", periodic=" << p.periodic_boundary
              << ", boundary_prefer=" << p.boundary_prefer
              << ", total_nodes=" << p.total_nodes_for_bitpacking
              << ", samples per rho=" << p.num_samples
              << ", max_threads=" << Max_thread_number
              << ", output dir base: " << dir << std::endl;

    char confirm;
    std::cout << "Confirm to start computation? (y/n)" << std::endl;
    std::cin >> confirm;
    if (!(confirm == 'y' || confirm == 'Y')) {
        std::cout << "Computation cancelled." << std::endl;
        system("pause");
        return -1;
    }

    // 保存原始主种子
    unsigned int original_seed = p.seed;

    // 对每个 rho 进行循环
    for (double current_rho = rho_start; current_rho <= rho_end + 1e-12; current_rho += rho_step) {
        p.rho = current_rho;
        p.seed = original_seed;

        std::cout << "\n--- Processing rho = " << p.rho << " ---" << std::endl;

        // 生成当前 rho 下的样本种子列表
        std::vector<uint32_t> seed_list = get_random_seed_list(p);

        // 构建当前 rho 的输出目录
        std::string rho_str = format_rho_string(p.rho);
        std::string boundary_str = p.periodic_boundary ? "-Periodic" : "-NonPeriodic";
        if (!p.periodic_boundary) boundary_str = boundary_str + "-" + p.boundary_prefer;
        std::string current_dir = dir + "/d=" + std::to_string(p.d) +
                                  "-L=" + std::to_string(p.L) + "-W=" + std::to_string(p.W) +
                                  boundary_str  + "-main_seed=" + std::to_string(p.seed)+ "/rho=" + rho_str;

        // 创建目录
    #if __has_include(<filesystem>)
            if (!fs::exists(current_dir)) fs::create_directories(current_dir);
    #else
            struct stat st;
            if (stat(current_dir.c_str(), &st) == -1) create_directories_compat(current_dir);
    #endif

        // 输出文件路径
        std::string filename = current_dir + "/Active_lattice_type.txt";
        std::string lcc_filename = current_dir + "/LCC_lattice_type.txt";

        // 写入文件头（覆盖模式）
        std::ofstream header1(filename);
        header1 << "sample_id, N_active, N_A_1, N_B_1, N_A_0, N_B_0, N_A_*, N_B_*, seed\n";
        header1.close();
        std::ofstream header2(lcc_filename);
        header2 << "sample_id, N_LCC, N_A_1_LCC, N_B_1_LCC, N_A_0_LCC, N_B_0_LCC, N_A_*_LCC, N_B_*_LCC, seed\n";
        header2.close();

        // 互斥锁
        std::mutex file1_mutex, file2_mutex;

        // 确定线程数
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = Max_thread_number;
        num_threads = std::min(num_threads, Max_thread_number);
        std::cout << "Using " << num_threads << " threads for " << p.num_samples << " samples." << std::endl;

        // 任务队列和完成计数器
        ThreadSafeQueue<SampleTask> task_queue;
        std::atomic<int> completed_count(0);

        // 填充任务
        for (int sample_id = 0; sample_id < p.num_samples; ++sample_id) {
            unsigned int sample_seed = seed_list[sample_id];
            SampleTask task;
            task.seed = sample_seed;
            task.sample_id = sample_id;
            task_queue.push(task);
        }

        // 启动工作线程
        std::vector<std::thread> threads;
        for (unsigned int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker_thread,
                                 std::ref(task_queue),
                                 std::cref(p),
                                 filename, lcc_filename,
                                 std::ref(file1_mutex), std::ref(file2_mutex),
                                 std::ref(completed_count),
                                 p.num_samples);
        }

        // 等待所有线程完成
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }

        std::cout << "Completed rho = " << p.rho << " (" << p.num_samples << " samples)" << std::endl;
    }

    std::cout << "\nAll rho values processed successfully!" << std::endl;
    return 0;
}