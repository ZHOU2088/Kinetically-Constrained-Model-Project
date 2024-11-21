#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <tuple>
#include <omp.h>

// g++ -O3 -fopenmp -march=native ComputeLargeComponent3.cpp -o ComputeLargeComponent
// ./ComputeLargeComponent


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

    LatticeProcessor(int l_x, int l_y, int k, double p0, int random_seed, std::string graph_type)
        : L_x(l_x), L_y(l_y), K(k), p_0(p0), seed(random_seed) {
        if (graph_type == "hexagon"){
            neighbors_list = get_neighbors_hexagon();}
        if (graph_type == "square"){
            neighbors_list = get_neighbors_square();}

        initialize_states();
    }

    double process() {
        compute_k_core();
        remain_flag = compute_remain_flag();
        auto new_neighbors_list = reindex_neighbors();

        double radio = static_cast<double>(new_neighbors_list.size()) / (L_x * L_y);
        return radio;
    }

private:
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
};

void save_result(std::ofstream& ofs, int L, double p_0, double mean, double variance) {
    if (!ofs.is_open()) {
        throw std::runtime_error("File stream is not open for writing.");
    }
    ofs << L << " " << p_0 << " " << mean << " " << variance << "\n";
}

int main() {
    std::vector<int> L_values = {200,400,600,800,1000};
    const double p0_start = 0.0;
    const double p0_end = 0.5;
    const double p0_step = 0.002;
    const int num_samples = 10000;
    const int K = 1;

    std::string graph_type = "square";



    std::filesystem::create_directory("plot_data");
    std::ofstream ofs("plot_data/rho-remain_component-"+graph_type+".txt");
    if (!ofs.is_open()) {
        throw std::runtime_error("plot_data/rho-remain_component-"+graph_type+".txt");
    }

    for (int L : L_values) {
        #pragma omp parallel for
        for (int idx = 0; idx < static_cast<int>((p0_end - p0_start) / p0_step) + 1; ++idx) {
            double p_0 = p0_start + idx * p0_step;
            std::vector<double> ratios;
            std::random_device rd;
            unsigned seed_base = rd();

            for (int i = 0; i < num_samples; ++i) {
                int seed = seed_base + i;
                LatticeProcessor lp(L, L, K, p_0, seed, graph_type);
                double ratio = lp.process();
                ratios.push_back(ratio);
            }

            double mean = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
            double variance = std::accumulate(ratios.begin(), ratios.end(), 0.0, [mean](double acc, double x) {
                return acc + (x - mean) * (x - mean);
            }) / ratios.size();

            #pragma omp critical
            {
                std::cout << "L = " << L << ", p_0 = " << p_0 
                          << ", Mean = " << mean 
                          << ", Variance = " << variance << std::endl;

                save_result(ofs, L, p_0, mean, variance);
            }
        }
    }

    ofs.close();
    return 0;
}

