#include <iostream> 
#include <fstream> 
#include <vector> 
#include <string> 
#include <random> 
#include <algorithm> 
#include <numeric> 
#include <chrono> 
#include <cmath> 
#include <stdexcept> 
#include <iterator> 
#include <sstream> 
#include <cstdint>  
#include <iomanip> 
#include <filesystem>
#include <queue>
#include <unordered_map>
#include <omp.h>



// g++ -O3 -fopenmp -march=native CLT2.cpp -o CLT
// ./CLT



class LatticeProcessor { 
public:
    int L_x; 
    int L_y; 
    int K; 
    double p_0; 
    int seed; 
    std::vector<std::vector<int>> neighbors_list; 
    std::vector<int> states; 
    std::vector<int> remain_flag; 
    size_t num_vertices;


    LatticeProcessor(int l_x, int l_y, int k, double p0, int random_seed, std::string graph_type) 
        : L_x(l_x), L_y(l_y), K(k), p_0(p0), seed(random_seed) {
        if (graph_type == "hexagon"){
            neighbors_list = get_neighbors_hexagon();}
        if (graph_type == "square"){
            neighbors_list = get_neighbors_square();}
        initialize_states(); 
        num_vertices = L_x * L_y;
    }


    
    void dfs(int node, std::vector<bool>& visited, std::vector<int>& component) {
        std::vector<int> stack{node}; 
        while (!stack.empty()) { 
            int n = stack.back(); 
            stack.pop_back(); 

            if (!visited[n]) { 
                visited[n] = true; 
                component.push_back(n); 
                for (int neighbor : neighbors_list[n]) { 
                    if (remain_flag[neighbor] == 1 && !visited[neighbor]) { 
                        stack.push_back(neighbor); 
                    }
                }
            }
        }
    }

    
    std::vector<int> find_largest_connected_component() {
        std::vector<bool> visited(remain_flag.size(), false); 
        std::vector<int> largest_component; 

        for (int i = 0; i < remain_flag.size(); ++i) { 
            if (remain_flag[i] == 1 && !visited[i]) { 
                std::vector<int> current_component; 
                dfs(i, visited, current_component); 
                if (current_component.size() > largest_component.size()) { 
                    largest_component = current_component;
                }
            }
        }

        std::vector<int> new_remain_flag(remain_flag.size(), 0); 
        for (int node : largest_component) {
            new_remain_flag[node] = 1; 
        }

        return new_remain_flag; 
    }

    
    std::vector<int> compute_remain_flag() {
        remain_flag = std::vector<int>(L_x * L_y, 1); 

        
        for (int i = 0; i < neighbors_list.size(); ++i) {
            int active_neighbors = 0; 
            for (int n : neighbors_list[i]) {
                active_neighbors += states[n]; 
            }
            if (active_neighbors >= K) {
                remain_flag[i] = 0; 
            }
        }

        
        for (int i = 0; i < neighbors_list.size(); ++i) {
            int active_neighbors = 0; 
            for (int n : neighbors_list[i]) {
                active_neighbors += remain_flag[n]; 
            }
            if (active_neighbors == 0) {
                remain_flag[i] = 0; 
            }
        }

        remain_flag = find_largest_connected_component(); 
        return remain_flag; 
    }


    
    std::vector<std::vector<int>> get_neighbors_hexagon() {
        std::vector<std::vector<int>> neighbors; 
        neighbors.reserve(L_x * L_y); 

        for (int y = 0; y < L_y; ++y) { 
            for (int x = 0; x < L_x; ++x) { 
                std::vector<std::pair<int, int>> neighbors_xy; 
                
                if (y % 2 == 0) { 
                    neighbors_xy = {
                        {(x + 1) % L_x, y}, 
                        {(x - 1 + L_x) % L_x, y}, 
                        {x, (y + 1) % L_y}, 
                        {x, (y - 1 + L_y) % L_y}, 
                        {(x - 1 + L_x) % L_x, (y - 1 + L_y) % L_y}, 
                        {(x - 1 + L_x) % L_x, (y + 1) % L_y} 
                    };
                } else { 
                    neighbors_xy = {
                        {(x + 1) % L_x, y}, 
                        {(x - 1 + L_x) % L_x, y}, 
                        {x, (y + 1) % L_y}, 
                        {x, (y - 1 + L_y) % L_y}, 
                        {(x + 1) % L_x, (y - 1 + L_y) % L_y}, 
                        {(x + 1) % L_x, (y + 1) % L_y} 
                    };
                }

                std::vector<int> neighbors_index; 
                for (const auto& n : neighbors_xy) {
                    neighbors_index.push_back(n.first + L_x * n.second); 
                }
                neighbors.push_back(neighbors_index); 
            }
        }
        return neighbors; 
    }



    std::vector<std::vector<int>> get_neighbors_square() {
        std::vector<std::vector<int>> neighbors;
        neighbors.reserve(L_x * L_y);

        for (int y = 0; y < L_y; ++y) {
            for (int x = 0; x < L_x; ++x) {
                // 对于正方晶格，每个点都有相同的四个最近邻
                std::vector<std::pair<int, int>> neighbors_xy = {
                    {(x + 1) % L_x, y},           // 右邻居
                    {(x - 1 + L_x) % L_x, y},     // 左邻居
                    {x, (y + 1) % L_y},           // 上邻居
                    {x, (y - 1 + L_y) % L_y}      // 下邻居
                };

                // 将二维坐标转换为一维索引
                std::vector<int> neighbors_index;
                neighbors_index.reserve(4);  // 正方晶格每个点有4个最近邻
                for (const auto& n : neighbors_xy) {
                    neighbors_index.push_back(n.first + L_x * n.second);
                }
                neighbors.push_back(neighbors_index);
            }
        }
        return neighbors;
    }


    
    void initialize_states() {
        std::mt19937 gen(seed); 
        std::uniform_real_distribution<> dis(0.0, 1.0); 
        states.resize(L_x * L_y); 
        for (int i = 0; i < L_x * L_y; ++i) {
            states[i] = (dis(gen) < p_0) ? 1 : 0; 
        }
    }

    
    void compute_k_core() {
        bool changed = true; 
        while (changed) { 
            changed = false; 
            for (int i = 0; i < neighbors_list.size(); ++i) { 
                if (states[i] == 1) { 
                    int active_neighbors = 0; 
                    for (int n : neighbors_list[i]) {
                        active_neighbors += states[n]; 
                    }
                    if (active_neighbors < K) { 
                        states[i] = 0; 
                        changed = true; 
                    }
                }
            }
        }
    }

    
    std::vector<std::vector<int>> reindex_neighbors() {
        std::unordered_map<int, int> index_map; 
        int new_index = 0; 
        
        for (int i = 0; i < states.size(); ++i) {
            if (remain_flag[i] == 1) { 
                index_map[i] = new_index++; 
            }
        }

        std::vector<std::vector<int>> new_neighbors_list; 
        for (int i = 0; i < states.size(); ++i) {
            if (remain_flag[i] == 1) { 
                std::vector<int> new_neighbors; 
                for (int n : neighbors_list[i]) {
                    if (remain_flag[n] == 1) { 
                        new_neighbors.push_back(index_map[n]); 
                    }
                }
                new_neighbors_list.push_back(new_neighbors); 
            }
        }
        
        return new_neighbors_list; 
    }

    
    
    std::vector<std::vector<int>>  process() {
        compute_k_core(); 
        remain_flag = compute_remain_flag(); 

        auto new_neighbors_list = reindex_neighbors(); 

        return new_neighbors_list; 
    }

    // 按照 remain_flag 过滤 state 数组
    std::vector<int> filter_state(const std::vector<int>& state) {
        std::vector<int> filtered_state;
        for (size_t i = 0; i < state.size(); ++i) {
            if (remain_flag[i] == 1) {
                filtered_state.push_back(state[i]);
            }
        }
        return filtered_state;
    }

    std::vector<uint64_t> compress_to_uint64(const std::vector<int>& state) {
        std::vector<uint64_t> compressed;
        size_t size = state.size();
        size_t num_uint64 = (size + 63) / 64; // Calculate number of uint64_t needed

        for (size_t i = 0; i < num_uint64; ++i) {
            uint64_t value = 0;
            for (size_t bit = 0; bit < 64; ++bit) {
                size_t index = i * 64 + bit;
                if (index < size && state[index] == 1) {
                    value |= (1ULL << bit);
                }
            }
            compressed.push_back(value);
        }

        return compressed;
    }

    std::vector<std::vector<uint64_t>> create_hard_hexagon_ground_states() {
        // Assume state1, state2, and state3 are already defined and populated
        std::vector<int> state1(L_y * L_x, 0);
        std::vector<int> state2(L_y * L_x, 0);
        std::vector<int> state3(L_y * L_x, 0);

        // Populate state1, state2, and state3 as per your logic
        for (int y = 0; y < L_y; ++y) {
            for (int x = 0; x < L_x; ++x) {
                int index = y * L_x + x;
                if (y % 2 == 0) {
                    if (x % 3 == 0) {
                        state1[index] = 1;
                    } else if (x % 3 == 1) {
                        state2[index] = 1;
                    } else if (x % 3 == 2) {
                        state3[index] = 1;
                    }
                } else {
                    if (x % 3 == 1) {
                        state1[index] = 1;
                    } else if (x % 3 == 2) {
                        state2[index] = 1;
                    } else if (x % 3 == 0) {
                        state3[index] = 1;
                    }
                }
            }
        }

        // Filter the states based on remain_flag
        std::vector<int> filtered_state1 = filter_state(state1);
        std::vector<int> filtered_state2 = filter_state(state2);
        std::vector<int> filtered_state3 = filter_state(state3);

        // Compress filtered states
        std::vector<std::vector<uint64_t>> compressed_states;
        compressed_states.push_back(compress_to_uint64(filtered_state1));
        compressed_states.push_back(compress_to_uint64(filtered_state2));
        compressed_states.push_back(compress_to_uint64(filtered_state3));

        return compressed_states;
    }



    std::vector<double> calculate_ground_state_density() {
        // 生成三个基态
        std::vector<int> state1(L_y * L_x, 0);
        std::vector<int> state2(L_y * L_x, 0);
        std::vector<int> state3(L_y * L_x, 0);

        // 填充三个基态
        for (int y = 0; y < L_y; ++y) {
            for (int x = 0; x < L_x; ++x) {
                int index = y * L_x + x;
                if (y % 2 == 0) {
                    if (x % 3 == 0) state1[index] = 1;
                    else if (x % 3 == 1) state2[index] = 1;
                    else if (x % 3 == 2) state3[index] = 1;
                } else {
                    if (x % 3 == 1) state1[index] = 1;
                    else if (x % 3 == 2) state2[index] = 1;
                    else if (x % 3 == 0) state3[index] = 1;
                }
            }
        }

        // 计算每个基态与remain_flag的内积
        int dot1 = 0, dot2 = 0, dot3 = 0;

        for (int i = 0; i < L_x * L_y; ++i) {
            dot1 += state1[i] * remain_flag[i];
            dot2 += state2[i] * remain_flag[i];
            dot3 += state3[i] * remain_flag[i];
        }

        std::vector<double> ground_state_density(3);

        num_vertices = std::count(remain_flag.begin(), remain_flag.end(), 1); 

        ground_state_density[0] = static_cast<double>(dot1) / num_vertices;
        ground_state_density[1] = static_cast<double>(dot2) / num_vertices;
        ground_state_density[2] = static_cast<double>(dot3) / num_vertices;

        return ground_state_density;
    }

    std::vector<int> calculate_ground_state_energies() {
        // 初始化 ground_state 向量，用于存储三个基态的密度，每个元素初始化为 0
        std::vector<int> ground_state(3, 0);

        // 遍历二维网格的行（y 方向）
        for (int y = 0; y < L_y; ++y) {
            // 遍历二维网格的列（x 方向）
            for (int x = 0; x < L_x; ++x) {
                // 计算当前坐标在一维数组中的索引
                int index = y * L_x + x;
                int state_index;

                // 根据 y 的奇偶性来决定 x 的基态分布
                if (y % 2 == 0) {
                    // 当 y 是偶数行时，x 对 3 取模直接决定基态索引
                    state_index = x % 3;
                } else {
                    // 当 y 是奇数行时，将 x + 2 后对 3 取模来决定基态索引
                    state_index = (x + 2) % 3;
                }

                // 若 remain_flag 在当前索引位置为 1，则在相应的基态累加
                if (remain_flag[index] == 1) {
                    ground_state[state_index]++;
                }
            }
        }

        // 返回三个基态的密度
        return ground_state;
    }




};



double ComputeOverlap01(const std::vector<uint64_t>& state_1, const std::vector<uint64_t>& state_2) {
    double q_t = 0.0;
    int count_state2 = 0;

    for (size_t j = 0; j < state_1.size(); ++j) {
        uint64_t and_result = state_1[j] & state_2[j];
        q_t += __builtin_popcountll(and_result);
        count_state2 += __builtin_popcountll(state_2[j]);
    }

    if (count_state2 == 0) {
        return 0.0;
    }

    q_t /= count_state2;
    return q_t;
}


std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result;
    if (num <= 0) return result;
    if (num == 1) {
        result.push_back(start);
        return result;
    }
    
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(start + i * step);
    }
    return result;
}



int main() {
    // 初始化固定参数
    int graph_seed = 114514;
    int threshold = 1;
    int num_samples = 1e5;
    std::string graph_type = "square";

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 遍历不同的 L_x 和 p_0
    std::vector<int> L_x_values = {800};  // 示例值，可以调整
    std::vector<double> p_0_values = linspace(0.01,0.5,50);  // 示例值，可以调整


    // 检测和打印系统支持的最大线程数
    int max_threads = omp_get_max_threads();
    std::cout << "Maximum available threads: " << max_threads << std::endl;

    // 检测和打印可用的处理器数量
    int num_procs = omp_get_num_procs();
    std::cout << "Number of processors available: " << num_procs << std::endl;

    // 以此值设置 OpenMP 使用的线程数（如果需要）
    omp_set_num_threads(max_threads);



    for (int L_x : L_x_values) {
        int L_y = L_x;  // 假设 L_y 与 L_x 相同
        
        for (double p_0 : p_0_values) {
            std::cout << "L_x = " << L_x << ", p_0 = " << p_0 << std::endl;
            // 设置输出文件名
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << p_0;
            std::string p0_str = oss.str();
            p0_str.erase(p0_str.find_last_not_of('0') + 1, std::string::npos);
            if (p0_str.back() == '.') {
                p0_str.pop_back();
            }
            if (p_0 == 0) {
                p0_str = "0";
            }
            std::string RandomGraphType = graph_type+"Lattice-" + p0_str + "-" + std::to_string(L_x * L_y);
            std::string filename = "./CLT6_data/ground_state_density_" + RandomGraphType + ".txt";
            
            std::filesystem::create_directories("./CLT6_data");
            std::ofstream outfile(filename);
            if (!outfile) {
                std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
                continue;  // 跳过当前组合，继续下一个
            }

            // 存储密度向量
            // std::vector<double> density1(num_samples), density2(num_samples), density3(num_samples);
            std::vector<int> energy1(num_samples), energy2(num_samples), energy3(num_samples);

            #pragma omp parallel for
            // 并行化采样
            for (int i = 0; i < num_samples; ++i) {
                LatticeProcessor processor(L_x, L_y, threshold, p_0, graph_seed + i,graph_type);
                processor.process();
                // std::vector<double> densities = processor.calculate_ground_state_density();

                std::vector<int>   energies = processor.calculate_ground_state_energies();
                
                energy1[i] = energies[0];
                energy2[i] = energies[1];
                energy3[i] = energies[2];
            }

            // 将每种密度写入文件，每种密度一个完整的行
            auto write_density_line = [&outfile](const std::vector<int>& densities) {
                for (int density : densities) {
                    outfile << density << " ";
                }
                outfile << "\n";
            };

            write_density_line(energy1);
            write_density_line(energy2);
            write_density_line(energy3);

            outfile.close();
            std::cout << "Results saved to " << filename << std::endl;
        }
    }

    // 结束计时并输出执行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Total time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}