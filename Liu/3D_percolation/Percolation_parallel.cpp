#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>

// 三维向量结构
struct Vec3 {
    int x, y, z;
    Vec3(int x, int y, int z) : x(x), y(y), z(z) {}
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// 哈希函数用于Vec3
struct Vec3Hash {
    size_t operator()(const Vec3& v) const {
        return ((std::hash<int>()(v.x) ^ (std::hash<int>()(v.y) << 1)) >> 1) ^ 
               (std::hash<int>()(v.z) << 1);
    }
};

// 三维晶格类
class Lattice3D {
private:
    int L;
    std::vector<int8_t> data;
    
public:
    Lattice3D(int size) : L(size), data(size * size * size, 0) {}
    
    // 获取/设置格点值
    int8_t& operator()(int x, int y, int z) {
        return data[z * L * L + y * L + x];
    }
    
    const int8_t& operator()(int x, int y, int z) const {
        return data[z * L * L + y * L + x];
    }
    
    int size() const { return L; }
    int total_sites() const { return L * L * L; }
    
    // 随机初始化
    void random_init(double prob, bool use_p, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (int z = 0; z < L; ++z) {
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    double r = dist(rng);
                    if (use_p) {
                        // p是占据概率：random_val < p -> 1，否则0
                        (*this)(x, y, z) = (r < prob) ? 1 : 0;
                    } else {
                        // rho是空置概率：random_val > rho -> 1，否则0
                        (*this)(x, y, z) = (r > prob) ? 1 : 0;
                    }
                }
            }
        }
    }
};

// 并查集类用于霍森科佩尔曼算法
class UnionFind {
private:
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    UnionFind(int size) : parent(size), rank(size, 0) {
        for (int i = 0; i < size; ++i) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        // 路径压缩
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }
    
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        
        if (rootX != rootY) {
            // 按秩合并
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
    
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};
// 团簇标记算法枚举
enum class ClusterAlgorithm {
    BFS,
    HOSHEN_KOPPELMAN
};

// 邻居方向定义
const std::vector<Vec3> NEIGHBOR_DIRECTIONS = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
};

// BFS算法实现团簇标记（支持周期性和非周期性边界条件）
std::pair<std::vector<int>, int> label_clusters_bfs(const Lattice3D& lattice, bool periodic) {
    int L = lattice.size();
    int total_sites = lattice.total_sites();
    
    std::vector<int> labels(total_sites, 0);
    std::vector<bool> visited(total_sites, false);
    int current_label = 1;
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = z * L * L + y * L + x;
                
                if (lattice(x, y, z) == 1 && !visited[idx]) {
                    // BFS遍历当前团簇
                    std::queue<Vec3> q;
                    q.push({x, y, z});
                    visited[idx] = true;
                    labels[idx] = current_label;
                    
                    while (!q.empty()) {
                        Vec3 current = q.front();
                        q.pop();
                        
                        for (const auto& dir : NEIGHBOR_DIRECTIONS) {
                            int nx, ny, nz;
                            
                            if (periodic) {
                                // 周期性边界条件
                                nx = (current.x + dir.x + L) % L;
                                ny = (current.y + dir.y + L) % L;
                                nz = (current.z + dir.z + L) % L;
                            } else {
                                // 非周期性边界条件
                                nx = current.x + dir.x;
                                ny = current.y + dir.y;
                                nz = current.z + dir.z;
                                
                                // 检查边界
                                if (nx < 0 || nx >= L || ny < 0 || ny >= L || nz < 0 || nz >= L) {
                                    continue;
                                }
                            }
                            
                            int nidx = nz * L * L + ny * L + nx;
                            
                            if (lattice(nx, ny, nz) == 1 && !visited[nidx]) {
                                visited[nidx] = true;
                                labels[nidx] = current_label;
                                q.push({nx, ny, nz});
                            }
                        }
                    }
                    
                    current_label++;
                }
            }
        }
    }
    
    return {labels, current_label - 1};
}

// 霍森科佩尔曼算法实现团簇标记（支持周期性和非周期性边界条件）
std::pair<std::vector<int>, int> label_clusters_hoshen_koppelmann(const Lattice3D& lattice, bool periodic) {
    int L = lattice.size();
    int total_sites = lattice.total_sites();
    
    UnionFind uf(total_sites);
    std::vector<int> top_label(L * L, 0);  // 存储上一层格点的标签
    
    int current_label = 1;
    
    // 第一遍扫描：合并连通分量
    for (int z = 0; z < L; ++z) {
        std::vector<int> current_label_map(L * L, 0);
        
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                if (lattice(x, y, z) == 1) {
                    int idx = z * L * L + y * L + x;
                    int label = 0;
                    
                    // 检查左侧邻居（同一层）
                    if (x > 0 && lattice(x-1, y, z) == 1) {
                        int left_idx = z * L * L + y * L + (x-1);
                        label = uf.find(left_idx);
                        current_label_map[y * L + x] = label;
                    }
                    
                    // 检查上层邻居（上一层的同一位置）
                    if (z > 0 && top_label[y * L + x] != 0) {
                        int top_label_val = top_label[y * L + x];
                        if (label == 0) {
                            label = top_label_val;
                            current_label_map[y * L + x] = label;
                        } else {
                            // 合并当前层左侧和上层的标签
                            uf.unite(label, top_label_val);
                        }
                    }
                    
                    // 检查前向邻居（同一层，y方向）
                    if (y > 0 && lattice(x, y-1, z) == 1) {
                        int forward_idx = z * L * L + (y-1) * L + x;
                        int forward_label = uf.find(forward_idx);
                        if (label == 0) {
                            label = forward_label;
                            current_label_map[y * L + x] = label;
                        } else {
                            uf.unite(label, forward_label);
                        }
                    }
                    
                    // 如果没有邻居，创建新标签
                    if (label == 0) {
                        current_label_map[y * L + x] = current_label;
                        // 需要将current_label映射到一个实际的索引
                        // 这里我们暂时使用idx作为父节点
                        // 在UnionFind中，我们使用label值作为索引
                        // 但需要保证label值在范围内
                        current_label++;
                    }
                }
            }
        }
        
        // 更新top_label为当前层的标签
        top_label = current_label_map;
    }
    
    // 第二遍扫描：重新编号标签
    std::vector<int> labels(total_sites, 0);
    std::unordered_map<int, int> root_to_label;
    int final_label = 1;
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                if (lattice(x, y, z) == 1) {
                    int idx = z * L * L + y * L + x;
                    int root = uf.find(idx);
                    
                    if (root_to_label.find(root) == root_to_label.end()) {
                        root_to_label[root] = final_label++;
                    }
                    labels[idx] = root_to_label[root];
                }
            }
        }
    }
    
    return {labels, final_label - 1};
}

// 改进的霍森科佩尔曼算法（更通用的实现）
std::pair<std::vector<int>, int> label_clusters_hoshen_koppelmann_improved(const Lattice3D& lattice, bool periodic) {
    int L = lattice.size();
    int total_sites = lattice.total_sites();
    
    UnionFind uf(total_sites);
    
    // 第一遍：扫描所有格点并合并连通分量
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                if (lattice(x, y, z) == 1) {
                    int idx = z * L * L + y * L + x;
                    
                    // 检查前向和左侧邻居（同一层）
                    if (x > 0 && lattice(x-1, y, z) == 1) {
                        int left_idx = z * L * L + y * L + (x-1);
                        uf.unite(idx, left_idx);
                    }
                    
                    if (y > 0 && lattice(x, y-1, z) == 1) {
                        int forward_idx = z * L * L + (y-1) * L + x;
                        uf.unite(idx, forward_idx);
                    }
                    
                    // 检查上层邻居
                    if (z > 0 && lattice(x, y, z-1) == 1) {
                        int up_idx = (z-1) * L * L + y * L + x;
                        uf.unite(idx, up_idx);
                    }
                    
                    // 如果是周期性边界条件，还需要检查边界连接
                    if (periodic) {
                        // 检查x方向边界
                        if (x == 0 && lattice(L-1, y, z) == 1) {
                            int boundary_idx = z * L * L + y * L + (L-1);
                            uf.unite(idx, boundary_idx);
                        }
                        
                        // 检查y方向边界
                        if (y == 0 && lattice(x, L-1, z) == 1) {
                            int boundary_idx = z * L * L + (L-1) * L + x;
                            uf.unite(idx, boundary_idx);
                        }
                        
                        // 检查z方向边界
                        if (z == 0 && lattice(x, y, L-1) == 1) {
                            int boundary_idx = (L-1) * L * L + y * L + x;
                            uf.unite(idx, boundary_idx);
                        }
                    }
                }
            }
        }
    }
    
    // 第二遍：重新编号标签
    std::vector<int> labels(total_sites, 0);
    std::unordered_map<int, int> root_to_label;
    int current_label = 1;
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                if (lattice(x, y, z) == 1) {
                    int idx = z * L * L + y * L + x;
                    int root = uf.find(idx);
                    
                    if (root_to_label.find(root) == root_to_label.end()) {
                        root_to_label[root] = current_label++;
                    }
                    labels[idx] = root_to_label[root];
                }
            }
        }
    }
    
    return {labels, current_label - 1};
}

// 修正的霍森科佩尔曼算法实现
std::pair<std::vector<int>, int> label_clusters_hoshen_koppelmann_corrected(const Lattice3D& lattice, bool periodic) {
    int L = lattice.size();
    int total_sites = lattice.total_sites();
    
    // 并查集使用格点索引作为元素
    UnionFind uf(total_sites);
    std::vector<int> labels(total_sites, 0);  // 存储每个格点的标签
    
    // 第一遍扫描：分配标签并合并连通分量
    int next_label = 1;
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = z * L * L + y * L + x;
                
                if (lattice(x, y, z) == 1) {
                    // 查找邻居中已分配标签的最小值
                    int min_neighbor_label = 0;
                    
                    // 检查左侧邻居（同一层）
                    if (x > 0 && lattice(x-1, y, z) == 1) {
                        int left_idx = z * L * L + y * L + (x-1);
                        if (labels[left_idx] > 0) {
                            if (min_neighbor_label == 0 || labels[left_idx] < min_neighbor_label) {
                                min_neighbor_label = labels[left_idx];
                            }
                        }
                    }
                    
                    // 检查前向邻居（同一层，y方向）
                    if (y > 0 && lattice(x, y-1, z) == 1) {
                        int forward_idx = z * L * L + (y-1) * L + x;
                        if (labels[forward_idx] > 0) {
                            if (min_neighbor_label == 0 || labels[forward_idx] < min_neighbor_label) {
                                min_neighbor_label = labels[forward_idx];
                            }
                        }
                    }
                    
                    // 检查上层邻居
                    if (z > 0 && lattice(x, y, z-1) == 1) {
                        int up_idx = (z-1) * L * L + y * L + x;
                        if (labels[up_idx] > 0) {
                            if (min_neighbor_label == 0 || labels[up_idx] < min_neighbor_label) {
                                min_neighbor_label = labels[up_idx];
                            }
                        }
                    }
                    
                    // 如果是周期性边界条件，还需要检查边界连接
                    if (periodic) {
                        // 检查x方向边界（最左侧）
                        if (x == 0 && lattice(L-1, y, z) == 1) {
                            int boundary_idx = z * L * L + y * L + (L-1);
                            if (labels[boundary_idx] > 0) {
                                if (min_neighbor_label == 0 || labels[boundary_idx] < min_neighbor_label) {
                                    min_neighbor_label = labels[boundary_idx];
                                }
                            }
                        }
                        
                        // 检查y方向边界（最前侧）
                        if (y == 0 && lattice(x, L-1, z) == 1) {
                            int boundary_idx = z * L * L + (L-1) * L + x;
                            if (labels[boundary_idx] > 0) {
                                if (min_neighbor_label == 0 || labels[boundary_idx] < min_neighbor_label) {
                                    min_neighbor_label = labels[boundary_idx];
                                }
                            }
                        }
                        
                        // 检查z方向边界（最上层）
                        if (z == 0 && lattice(x, y, L-1) == 1) {
                            int boundary_idx = (L-1) * L * L + y * L + x;
                            if (labels[boundary_idx] > 0) {
                                if (min_neighbor_label == 0 || labels[boundary_idx] < min_neighbor_label) {
                                    min_neighbor_label = labels[boundary_idx];
                                }
                            }
                        }
                    }
                    
                    if (min_neighbor_label > 0) {
                        // 有邻居已分配标签，使用最小标签
                        labels[idx] = min_neighbor_label;
                        
                        // 合并所有有标签的邻居
                        if (x > 0 && lattice(x-1, y, z) == 1 && labels[z * L * L + y * L + (x-1)] > 0) {
                            uf.unite(idx, z * L * L + y * L + (x-1));
                        }
                        if (y > 0 && lattice(x, y-1, z) == 1 && labels[z * L * L + (y-1) * L + x] > 0) {
                            uf.unite(idx, z * L * L + (y-1) * L + x);
                        }
                        if (z > 0 && lattice(x, y, z-1) == 1 && labels[(z-1) * L * L + y * L + x] > 0) {
                            uf.unite(idx, (z-1) * L * L + y * L + x);
                        }
                        
                        // 周期性边界条件的合并
                        if (periodic) {
                            if (x == 0 && lattice(L-1, y, z) == 1 && labels[z * L * L + y * L + (L-1)] > 0) {
                                uf.unite(idx, z * L * L + y * L + (L-1));
                            }
                            if (y == 0 && lattice(x, L-1, z) == 1 && labels[z * L * L + (L-1) * L + x] > 0) {
                                uf.unite(idx, z * L * L + (L-1) * L + x);
                            }
                            if (z == 0 && lattice(x, y, L-1) == 1 && labels[(L-1) * L * L + y * L + x] > 0) {
                                uf.unite(idx, (L-1) * L * L + y * L + x);
                            }
                        }
                    } else {
                        // 没有邻居有标签，分配新标签
                        labels[idx] = next_label++;
                    }
                }
            }
        }
    }
    
    // 第二遍扫描：重新编号标签，使用并查集的根
    std::unordered_map<int, int> root_to_label;
    int final_label = 1;
    std::vector<int> final_labels(total_sites, 0);
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                int idx = z * L * L + y * L + x;
                
                if (lattice(x, y, z) == 1) {
                    int root = uf.find(idx);
                    
                    // 如果这个根还没有标签，分配一个新标签
                    if (root_to_label.find(root) == root_to_label.end()) {
                        root_to_label[root] = final_label++;
                    }
                    
                    final_labels[idx] = root_to_label[root];
                }
            }
        }
    }
    
    return {final_labels, final_label - 1};  // 返回最终标签和团簇数量
}

// 通用的团簇标记函数，根据算法选择调用相应的实现
std::pair<std::vector<int>, int> label_clusters(const Lattice3D& lattice, bool periodic, ClusterAlgorithm algorithm) {
    switch (algorithm) {
        case ClusterAlgorithm::BFS:
            return label_clusters_bfs(lattice, periodic);
        case ClusterAlgorithm::HOSHEN_KOPPELMAN:
            return label_clusters_hoshen_koppelmann_corrected(lattice, periodic);
        default:
            std::cerr << "未知的团簇标记算法，使用BFS作为默认" << std::endl;
            return label_clusters_bfs(lattice, periodic);
    }
}

// 规则1应用函数
void apply_rule1(Lattice3D& intermediate, const Lattice3D& initial, bool periodic) {
    int L = initial.size();
    
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                intermediate(x, y, z) = initial(x, y, z);
                
                if (initial(x, y, z) == 0) {
                    bool all_neighbors_one = true;
                    int valid_neighbor_count = 0;
                    
                    for (const auto& dir : NEIGHBOR_DIRECTIONS) {
                        int nx, ny, nz;
                        
                        if (periodic) {
                            nx = (x + dir.x + L) % L;
                            ny = (y + dir.y + L) % L;
                            nz = (z + dir.z + L) % L;
                            valid_neighbor_count++;
                        } else {
                            nx = x + dir.x;
                            ny = y + dir.y;
                            nz = z + dir.z;
                            
                            if (nx >= 0 && nx < L && ny >= 0 && ny < L && nz >= 0 && nz < L) {
                                valid_neighbor_count++;
                            } else {
                                continue;
                            }
                        }
                        
                        if (initial(nx, ny, nz) == 0) {
                            all_neighbors_one = false;
                            break;
                        }
                    }
                    
                    if (valid_neighbor_count > 0 && all_neighbors_one) {
                        intermediate(x, y, z) = 1;
                    }
                }
            }
        }
    }
}

// 规则2应用函数（修正版）
void apply_rule2(Lattice3D& current_state, bool periodic) {
    int L = current_state.size();
    
    // 使用无序集合记录需要置0的格点，避免重复
    std::unordered_set<int> sites_to_zero;
    
    // 第一遍：找出所有需要置0的格点
    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                if (current_state(x, y, z) == 0) {
                    // 当前格点为0，将其所有邻居标记为需要置0
                    for (const auto& dir : NEIGHBOR_DIRECTIONS) {
                        int nx, ny, nz;
                        
                        if (periodic) {
                            nx = (x + dir.x + L) % L;
                            ny = (y + dir.y + L) % L;
                            nz = (z + dir.z + L) % L;
                        } else {
                            nx = x + dir.x;
                            ny = y + dir.y;
                            nz = z + dir.z;
                            
                            if (nx < 0 || nx >= L || ny < 0 || ny >= L || nz < 0 || nz >= L) {
                                continue;
                            }
                        }
                        
                        // 将邻居格点索引加入集合
                        int idx = nz * L * L + ny * L + nx;
                        sites_to_zero.insert(idx);
                    }
                }
            }
        }
    }
    
    // 第二遍：将所有标记的格点置0
    for (int idx : sites_to_zero) {
        int x = idx % L;
        int y = (idx / L) % L;
        int z = idx / (L * L);
        current_state(x, y, z) = 0;
    }
}

// 计算最大团簇大小
int find_max_cluster_size(const std::vector<int>& labels, int num_clusters) {
    if (num_clusters == 0) return 0;
    
    std::vector<int> cluster_sizes(num_clusters + 1, 0);  // 索引从1开始
    
    for (int label : labels) {
        if (label > 0) {
            cluster_sizes[label]++;
        }
    }
    
    return *std::max_element(cluster_sizes.begin() + 1, cluster_sizes.end());
}

// 单个样本处理函数
double process_single_sample(int L, double prob_val, bool use_p, bool periodic, 
                            ClusterAlgorithm algorithm, unsigned int seed) {
    // 初始化随机数生成器
    std::mt19937 rng(seed);
    
    // 创建晶格
    Lattice3D initial(L);
    initial.random_init(prob_val, use_p, rng);
    
    Lattice3D intermediate(L);
    Lattice3D final_state(L);
    
    // 应用规则1
    apply_rule1(intermediate, initial, periodic);
    
    // 应用规则2
    final_state = intermediate;  // 复制中间状态
    apply_rule2(final_state, periodic);
    
    // 团簇标记
    auto labeling_result = label_clusters(final_state, periodic, algorithm);
    
    // 计算最大团簇大小和密度
    int max_cluster_size = find_max_cluster_size(labeling_result.first, labeling_result.second);
    double max_cluster_density = static_cast<double>(max_cluster_size) / initial.total_sites();
    
    return max_cluster_density;
}

// 线程工作函数
void worker_function(int thread_id, int L, double prob_val, bool use_p, bool periodic,
                   ClusterAlgorithm algorithm, int start_sample, int end_sample, 
                   unsigned int base_seed, std::vector<double>& results, 
                   std::mutex& results_mutex) {
    
    std::vector<double> local_results;  // 本地存储
    local_results.reserve(end_sample - start_sample);
    
    for (int sample = start_sample; sample < end_sample; ++sample) {
        unsigned int seed = base_seed
                   ^ (std::hash<std::thread::id>()(std::this_thread::get_id()) << 1)
                   ^ (sample * 9973);
        double density = process_single_sample(L, prob_val, use_p, periodic, algorithm, seed);
        local_results.push_back(density);
    }
    // 批量写入，只锁一次
    std::lock_guard<std::mutex> lock(results_mutex);
    results.insert(results.end(), local_results.begin(), local_results.end());
}

// 创建优化的概率列表
std::vector<double> create_optimized_probability_list(double rho_min, double rho_max,double center_prob,
                                                     int num_points, double concentration) {
    std::vector<double> probs;
    double width = std::max(std::abs(rho_min - center_prob), std::abs(rho_max - center_prob));
    
    
    for (int i = 0; i < num_points; ++i) {
        double t = -1.0 + 2.0 * i / (num_points - 1);
        double transformed = std::copysign(std::pow(std::abs(t), concentration), t);
        double prob = center_prob + transformed * (width);
        
        if (prob >= rho_min && prob <= rho_max) {
            // 去重并保留6位小数
            prob = std::round(prob * 1e6) / 1e6;
            if (std::find(probs.begin(), probs.end(), prob) == probs.end()) {
                probs.push_back(prob);
            }
        }
    }
    
    std::sort(probs.begin(), probs.end());
    return probs;
}

// 主模拟函数
void run_simulation_3d(const std::vector<int>& L_list, 
                      const std::vector<double>& prob_list,
                      bool use_p, bool periodic, ClusterAlgorithm algorithm,
                      int num_samples, const std::string& output_dir, 
                      int num_threads) {
    
    // 创建输出目录（简化版，实际使用时需要更完整的目录创建逻辑）
    std::string prob_type = use_p ? "p" : "rho";
    std::string algorithm_str = (algorithm == ClusterAlgorithm::BFS) ? "bfs" : "hoshen";
    std::string boundary_str = periodic ? "periodic" : "nonperiodic";
    
    std::string detail_filename = output_dir + "/max-cluster-density-3d-" + prob_type + 
                                  "-" + algorithm_str + "-" + boundary_str + ".txt";
    std::string stat_filename = output_dir + "/max-cluster-density-stat-3d-" + prob_type + 
                                "-" + algorithm_str + "-" + boundary_str + ".txt";
    
    // 写入文件头
    std::ofstream detail_file(detail_filename);
    if(!detail_file.is_open())
    {
        std::cerr<<"Can not open file " << detail_filename << std::endl;
        return;
    }
    std::ofstream stat_file(stat_filename);
    if(!stat_file.is_open())
    {
        std::cerr<<"Can not open file " << stat_filename << std::endl;
        return;
    }
    
    detail_file << "L," << prob_type << ",max_cluster_densities\n";
    stat_file << "L," << prob_type << ",mean_density,std_density,second_moment,fourth_moment,Binder_cumulant\n";
    
    detail_file.flush();
    stat_file.flush();
    
    std::cout << "Start 3D simulation ..." << std::endl;
    std::cout << "Algorithm: " << algorithm_str << std::endl;
    std::cout << "Boundary: " << boundary_str << std::endl;
    std::cout << "L list: ";
    for (int L : L_list) std::cout << L << " ";
    std::cout << std::endl;
    std::cout << "Probability list size: " << prob_list.size() << std::endl;
    std::cout << "Thread number: " << num_threads << std::endl;
    
    unsigned int base_seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    for (int L : L_list) {
        std::cout << "\n System length L = " << L << std::endl;
        
        for (double prob_val : prob_list) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::vector<double> densities;
            std::mutex densities_mutex;
            std::vector<std::thread> threads;
            
            // 计算每个线程处理的样本数
            int samples_per_thread = num_samples / num_threads;
            int remaining_samples = num_samples % num_threads;
            
            // 创建并启动线程
            for (int i = 0; i < num_threads; ++i) {
                int start = i * samples_per_thread + std::min(i, remaining_samples);
                int end = start + samples_per_thread + (i < remaining_samples ? 1 : 0);
                
                threads.emplace_back(worker_function, i, L, prob_val, use_p, periodic,
                                   algorithm, start, end, base_seed, std::ref(densities), 
                                   std::ref(densities_mutex));
            }
            
            // 等待所有线程完成
            for (auto& thread : threads) {
                thread.join();
            }
            
            // 计算统计量
            if (!densities.empty()) {
                double mean = 0.0, std_dev = 0.0, second_moment = 0.0, fourth_moment = 0.0;
                
                for (double d : densities) {
                    mean += d;
                    second_moment += d * d;
                    fourth_moment += d * d * d * d;
                }
                
                mean /= densities.size();
                second_moment /= densities.size();
                fourth_moment /= densities.size();
                
                // 计算标准差
                for (double d : densities) {
                    std_dev += (d - mean) * (d - mean);
                }
                std_dev = std::sqrt(std_dev / densities.size());
                
                double Binder_cumulant = 1.0 - (fourth_moment / (3.0 * second_moment * second_moment));
                
                // 写入结果
                detail_file << L << "," << std::fixed << std::setprecision(7) << prob_val << ",";
                for (size_t i = 0; i < densities.size(); ++i) {
                    detail_file << std::scientific << std::setprecision(10) << densities[i];
                    if (i < densities.size() - 1) detail_file << ",";
                }
                detail_file << "\n";
                detail_file.flush();
                
                stat_file << L << "," << std::fixed << std::setprecision(7) << prob_val << ","
                         << std::scientific << std::setprecision(10) << mean << ","
                         << std_dev << "," << second_moment << "," << fourth_moment << ","
                         << Binder_cumulant << "\n";
                stat_file.flush();
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                double duration_time = static_cast<double>(duration.count()) / 1000.0;
                std::cout << "  " << prob_type << " = " << std::fixed << std::setprecision(4) << prob_val
                         << " Finished, Using " << std::setprecision(3) << duration_time << "s, mean density: " 
                         << std::scientific << std::setprecision(6) << mean 
                         << ", sample number: " << densities.size() << std::endl;
            }
        }
    }
    
    detail_file.close();
    stat_file.close();
    std::cout << "\nSimulation finished!" << std::endl;
    std::cout << "Detailed data: " << detail_filename << std::endl;
    std::cout << "Statics data: " << stat_filename << std::endl;
}

//g++ -std=c++11 -O3 -pthread Percolation_parallel.cpp -o percolation_parallel

int main() {
    // 模拟参数
    std::vector<int> L_list = {260,280,300};  // 测试用较小的尺寸
    
    // 创建优化的概率列表
    auto prob_list = create_optimized_probability_list(0.0, 0.46, 0.28, 100, 2.0);
    
    bool use_p = false;                    // 使用rho（空置概率）
    bool periodic = false;                 // 使用非周期性边界条件
    ClusterAlgorithm algorithm = ClusterAlgorithm::HOSHEN_KOPPELMAN; // 使用霍森科佩尔曼算法
    // ClusterAlgorithm algorithm = ClusterAlgorithm::BFS; // 或使用BFS算法
    int num_samples = 1000;               // 总样本数
    int num_threads = 14;                  // 线程数
    std::string output_dir = "./MIS-3D-results";
    
    // 运行模拟
    run_simulation_3d(L_list, prob_list, use_p, periodic, algorithm, num_samples, output_dir, num_threads);
    
    return 0;
}
