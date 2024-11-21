#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include<filesystem>

// g++ -O3 -march=native MCPottsModel2.cpp -o MCPottsModel
// ./MCPottsModel


std::vector<std::vector<int>> initializeNeighbors_hexagon(int L_x,int L_y) {
    // 定义一个函数 initializeNeighbors，接受两个整数参数 L_x 和 L_y，返回一个二维整数向量。
    std::vector<std::vector<int>> neighbors; 
    // 声明一个二维整数向量 neighbors，用于存储每个格点的邻居索引。
    neighbors.reserve(L_x * L_y); 
    // 预留空间以避免动态分配内存，提高效率。总共预留 L_x * L_y 个元素的空间。

    for (int y = 0; y < L_y; ++y) { 
        // 遍历每一行，y 从 0 到 L_y-1。
        for (int x = 0; x < L_x; ++x) { 
            // 遍历每一列，x 从 0 到 L_x-1。
            std::vector<std::pair<int, int>> neighbors_xy; 
            // 声明一个向量 neighbors_xy，用于存储当前格点的邻居坐标对。

            if (y % 2 == 0) { 
                // 如果当前行 y 是偶数行。
                neighbors_xy = {
                    {(x + 1) % L_x, y}, 
                    // 右边邻居，考虑周期性边界条件。
                    {(x - 1 + L_x) % L_x, y}, 
                    // 左边邻居，考虑周期性边界条件。
                    {x, (y + 1) % L_y}, 
                    // 上边邻居，考虑周期性边界条件。
                    {x, (y - 1 + L_y) % L_y}, 
                    // 下边邻居，考虑周期性边界条件。
                    {(x - 1 + L_x) % L_x, (y - 1 + L_y) % L_y}, 
                    // 左下角邻居，考虑周期性边界条件。
                    {(x - 1 + L_x) % L_x, (y + 1) % L_y} 
                    // 左上角邻居，考虑周期性边界条件。
                };
            } else { 
                // 如果当前行 y 是奇数行。
                neighbors_xy = {
                    {(x + 1) % L_x, y}, 
                    // 右边邻居，考虑周期性边界条件。
                    {(x - 1 + L_x) % L_x, y}, 
                    // 左边邻居，考虑周期性边界条件。
                    {x, (y + 1) % L_y}, 
                    // 上边邻居，考虑周期性边界条件。
                    {x, (y - 1 + L_y) % L_y}, 
                    // 下边邻居，考虑周期性边界条件。
                    {(x + 1) % L_x, (y - 1 + L_y) % L_y}, 
                    // 右下角邻居，考虑周期性边界条件。
                    {(x + 1) % L_x, (y + 1) % L_y} 
                    // 右上角邻居，考虑周期性边界条件。
                };
            }

            std::vector<int> neighbors_index; 
            // 声明一个整数向量 neighbors_index，用于存储当前格点的邻居索引。
            for (const auto& n : neighbors_xy) {
                // 遍历当前格点的所有邻居坐标对。
                neighbors_index.push_back(n.first + L_x * n.second); 
                // 将邻居坐标转换为索引，并添加到 neighbors_index 向量中。
            }
            neighbors.push_back(neighbors_index); 
            // 将当前格点的邻居索引列表添加到 neighbors 向量中。
        }
    }
    return neighbors; 
    // 返回包含所有格点邻居信息的二维向量 neighbors。
}



std::vector<std::vector<int>> initializeNeighbors_square(int L_x, int L_y) {
    // 创建一个二维向量用于存储每个格点的邻居索引
    std::vector<std::vector<int>> neighbors;
    neighbors.reserve(L_x * L_y); // 预留空间以提高效率

    // 遍历每个格点
    for (int y = 0; y < L_y; ++y) {
        for (int x = 0; x < L_x; ++x) {
            // 存储当前格点的邻居坐标
            std::vector<std::pair<int, int>> neighbors_xy = {
                {(x + 1) % L_x, y},               // 右边邻居
                {(x - 1 + L_x) % L_x, y},         // 左边邻居
                {x, (y + 1) % L_y},               // 上边邻居
                {x, (y - 1 + L_y) % L_y}          // 下边邻居
            };

            // 将邻居坐标转换为索引
            std::vector<int> neighbors_index;
            for (const auto& n : neighbors_xy) {
                neighbors_index.push_back(n.first + L_x * n.second);
            }

            // 将当前格点的邻居索引列表添加到 neighbors 向量中
            neighbors.push_back(neighbors_index);
        }
    }

    return neighbors; // 返回包含所有格点邻居信息的二维向量
}



class PottsModel {
private:
    const int L_x, L_y, q;
    // L_x 和 L_y 是系统的尺寸，q 是自旋状态的数目。
    const double J, sigma_J, sigma_h;
    // J 是相互作用强度，sigma_J 和 sigma_h 是相互作用和外场的标准差。
    std::vector<int> spins;
    // 存储每个格点的自旋状态。
    std::vector<std::vector<int>> neighbors;
    // 存储每个格点的邻居索引。
    std::vector<std::vector<double>> J_ij;
    // 存储每对邻居之间的耦合常数。
    std::vector<std::vector<double>> h_field;
    // 存储每个格点的外场。
    std::mt19937 gen;
    // 随机数生成器。
    std::uniform_real_distribution<> uniform_dist;
    // 均匀分布，用于生成随机数。
    std::normal_distribution<> normal_dist_J;
    // 正态分布，用于生成耦合常数。
    std::normal_distribution<> normal_dist_h;
    // 正态分布，用于生成外场。
    std::string graph_type;

    // 在线计算统计量的辅助类
    class OnlineStats {
    private:
        double sum = 0.0;
        // 存储样本的总和。
        double sum2 = 0.0;
        // 存储样本平方的总和。
        double sum4 = 0.0;
        // 存储样本四次方的总和。
        long long count = 0;  // 改用long long避免溢出
        // 样本数量，使用 long long 避免溢出。
    public:
        void add(double x) {
            sum += x;
            // 增加样本的总和。
            sum2 += x * x;
            // 增加样本平方的总和。
            sum4 += x * x * x * x;
            // 增加样本四次方的总和。
            count++;
            // 增加样本数量。
        }
        double mean() const { return count > 0 ? sum / count : 0.0; }
        // 返回样本的均值。
        double moment2() const { return count > 0 ? sum2 / count : 0.0; }
        // 返回样本的二阶矩。
        double moment4() const { return count > 0 ? sum4 / count : 0.0; }
        // 返回样本的四阶矩。
    };

public:
    PottsModel(int L_x_, int L_y_, int q_, double J_, double sigma_J_, double sigma_h_, unsigned seed, std::string graph_type_) 
        : L_x(L_x_), L_y(L_y_), q(q_), J(J_), sigma_J(sigma_J_), sigma_h(sigma_h_),  graph_type(graph_type_),
          spins(L_x * L_y, 0),  // 初始化为0
          // 初始化自旋状态为0。
          gen(seed),
          // 使用给定的种子初始化随机数生成器。
          uniform_dist(0.0, 1.0),
          // 初始化均匀分布在[0, 1]范围内。
          normal_dist_J(J, sigma_J),
          // 初始化耦合常数的正态分布。
          normal_dist_h(0.0, sigma_h) {
          // 初始化外场的正态分布。
        
        initializeSystem();
        // 调用初始化系统函数。
    }

    void initializeSystem() {
        // 初始化自旋构型
        for (int i = 0; i < L_x * L_y; ++i) {
            spins[i] = static_cast<int>(uniform_dist(gen) * q);  // 确保在[0, q-1]范围内
            // 随机初始化自旋状态，确保在[0, q-1]范围内。
        }

        // 初始化相互作用
        initializeInteractions();

        // 初始化外场
        h_field.resize(L_x * L_y, std::vector<double>(q, 0.0));  // 预分配并初始化为0
        // 预分配外场的空间并初始化为0。
        for (int i = 0; i < L_x * L_y; ++i) {
            for (int k = 0; k < q; ++k) {
                h_field[i][k] = normal_dist_h(gen);
                // 为每个自旋状态生成一个正态分布的外场。
            }
        }
    }

    void initializeInteractions() {
        // 初始化neighbors
        if (graph_type == "square"){
        neighbors = initializeNeighbors_square(L_x, L_y);}
        if (graph_type == "hexagon"){
        neighbors = initializeNeighbors_hexagon(L_x, L_y);}


        // 初始化邻居列表。
        
        // 初始化J_ij，预分配空间
        J_ij.resize(L_x * L_y);
        // 预分配耦合常数的空间。
        for (int i = 0; i < L_x * L_y; ++i) {
            J_ij[i].resize(neighbors[i].size(), 0.0);  // 预分配并初始化为0
            // 为每个格点的邻居分配空间并初始化为0。
        }

        // 设置耦合常数
        for (int i = 0; i < L_x * L_y; ++i) {
            for (size_t j = 0; j < neighbors[i].size(); ++j) {
                int neighbor = neighbors[i][j];
                if (neighbor > i) {  // 只为i < neighbor的对生成新的耦合常数
                    double J_value = normal_dist_J(gen);
                    // 为每对邻居生成一个正态分布的耦合常数。
                    J_ij[i][j] = J_value;
                    
                    // 找到对称位置并设置相同的耦合常数
                    auto it = std::find(neighbors[neighbor].begin(), neighbors[neighbor].end(), i);
                    if (it != neighbors[neighbor].end()) {
                        int idx = std::distance(neighbors[neighbor].begin(), it);
                        J_ij[neighbor][idx] = J_value;
                        // 设置对称位置的耦合常数相同。
                    }
                }
            }
        }
    }

    double calculateDeltaE(int site, int new_spin) {
        if (site < 0 || site >= L_x * L_y || new_spin < 0 || new_spin >= q) {
            return std::numeric_limits<double>::infinity();  // 无效的变化
            // 如果输入无效，返回无穷大。
        }

        double deltaE = 0.0;
        // 初始化能量变化为0。
        int old_spin = spins[site];
        // 获取当前格点的旧自旋状态。
        
        // 计算相互作用能变化
        for (size_t j = 0; j < neighbors[site].size(); ++j) {
            int neighbor = neighbors[site][j];
            if (neighbor >= 0 && neighbor < L_x * L_y) {  // 边界检查
                deltaE -= J_ij[site][j] * (
                    (spins[neighbor] == new_spin ? 1.0 : 0.0) -
                    (spins[neighbor] == old_spin ? 1.0 : 0.0)
                );
                // 计算相互作用能的变化。
            }
        }
        
        // 计算外场能变化
        deltaE -= h_field[site][new_spin] - h_field[site][old_spin];
        // 计算外场能的变化。
        
        return deltaE;
        // 返回能量变化。
    }

    double calculateEnergy() {
        double energy = 0.0;
        // 初始化总能量为0。
        
        // 计算相互作用能
        for (int i = 0; i < L_x * L_y; ++i) {
            for (size_t j = 0; j < neighbors[i].size(); ++j) {
                int neighbor = neighbors[i][j];
                if (neighbor > i && neighbor < L_x * L_y) {  // 避免重复计算和边界检查
                    energy -= J_ij[i][j] * (spins[i] == spins[neighbor] ? 1.0 : 0.0);
                    // 计算相互作用能。
                }
            }
        }
        
        // 计算外场能
        for (int i = 0; i < L_x * L_y; ++i) {
            energy -= h_field[i][spins[i]];
            // 计算外场能。
        }
        
        return energy;
        // 返回总能量。
    }

    double calculateOrderParameter() {
        std::vector<int> counts(q, 0);
        // 初始化一个向量用于计数每种自旋状态的数量。
        for (int spin : spins) {
            if (spin >= 0 && spin < q) {  // 边界检查
                counts[spin]++;
                // 计数每种自旋状态的数量。
            }
        }
        
        double max_diff = 0.0;
        // 初始化最大差值为0。
        for (int i = 0; i < q; ++i) {
            for (int j = i + 1; j < q; ++j) {
                double diff = std::abs(static_cast<double>(counts[i] - counts[j]) / (L_x * L_y));
                max_diff = std::max(max_diff, diff);
                // 计算不同自旋状态占比的最大差值。
            }
        }
        return max_diff;
        // 返回序参量。
    }

    std::vector<double> calculateSpinFractions() {
        std::vector<double> fractions(q, 0.0);
        // 初始化一个向量用于存储每种自旋状态的占比。
        int total = 0;
        // 初始化总计数为0。
        for (int spin : spins) {
            if (spin >= 0 && spin < q) {  // 边界检查
                fractions[spin] += 1.0;
                total++;
                // 计算每种自旋状态的数量。
            }
        }
        // 归一化
        if (total > 0) {
            for (double& fraction : fractions) {
                fraction /= total;
                // 归一化每种自旋状态的占比。
            }
        }
        return fractions;
        // 返回自旋状态的占比。
    }

    void runSimulation(double beta_start, double beta_end, double beta_factor,
                      int warmup_steps, int measure_interval, int num_states,
                      int num_measurements, const std::string& prefix) {
        // 运行模拟，参数包括起始和结束的 beta，beta 的变化因子，热身步数，测量间隔，状态数量，测量次数和文件前缀。
        
        std::ofstream state_file(prefix + "_states.txt");
        // 打开状态文件。
        std::ofstream stats_file(prefix + "_stats.txt");
        // 打开统计量文件。
        
        if (!state_file.is_open() || !stats_file.is_open()) {
            throw std::runtime_error("Unable to open output files");
            // 如果文件无法打开，抛出异常。
        }

        for (double beta = beta_start; beta <= beta_end; beta *= beta_factor) {
            // 遍历不同的 beta 值。
            // 热化
            for (int step = 0; step < warmup_steps; ++step) {
                performMCStep(beta);
                // 执行热身步。
            }
            
            // 测量
            OnlineStats energy_stats, order_param_stats;
            // 初始化能量和序参量的统计量。
            std::vector<OnlineStats> spin_fraction_stats(q);
            // 初始化自旋占比的统计量。
            int states_saved = 0;
            // 初始化已保存状态的数量。
            
            for (int m = 0; m < num_measurements; ++m) {
                performMCStep(beta);
                // 执行蒙特卡洛步。
                
                if (m % measure_interval == 0) {
                    double energy = calculateEnergy() / (L_x * L_y);
                    // 计算能量密度。
                    double order_param = calculateOrderParameter();
                    // 计算序参量。
                    std::vector<double> fractions = calculateSpinFractions();
                    // 计算自旋占比。
                    
                    energy_stats.add(energy);
                    // 添加能量到统计量。
                    order_param_stats.add(order_param);
                    // 添加序参量到统计量。
                    for (int k = 0; k < q; ++k) {
                        spin_fraction_stats[k].add(fractions[k]);
                        // 添加自旋占比到统计量。
                    }
                    
                    // 保存状态
                    if (states_saved < num_states) {
                        state_file << std::setprecision(10) << beta << " ";
                        for (int spin : spins) {
                            state_file << spin << " ";
                        }
                        state_file << "\n";
                        states_saved++;
                        // 保存自旋状态到文件。
                    }
                }
            }
            
            // 保存统计量
            stats_file << beta << " "
                      << energy_stats.mean() << " " << energy_stats.moment2() << " ";
            // 保存能量的统计量。
            
            for (int k = 0; k < q; ++k) {
                stats_file << spin_fraction_stats[k].mean() << " "
                          << spin_fraction_stats[k].moment2() << " ";
                // 保存自旋占比的统计量。
            }
            
            stats_file << order_param_stats.mean() << " "
                      << order_param_stats.moment2() << " "
                      << order_param_stats.moment4() << "\n";
            // 保存序参量的统计量。
            
            // 打印当前beta的结果
            std::cout << std::setprecision(6) << "beta: " << beta
                      << " energy: " << energy_stats.mean()
                      << " fraction: " ;
            for (int k = 0; k < q; ++k) { 
                std::cout<<spin_fraction_stats[k].mean()<<", ";
                // 打印自旋占比。
            }

            std::cout<< " order_param: " << order_param_stats.mean() << std::endl;
            // 打印序参量。
        }
        
        state_file.close();
        // 关闭状态文件。
        stats_file.close();
        // 关闭统计量文件。
    }

private:
    void performMCStep(double beta) {
        for (int i = 0; i < L_x * L_y; ++i) {
            int site = static_cast<int>(uniform_dist(gen) * (L_x * L_y));
            // 随机选择一个格点。
            int new_spin = static_cast<int>(uniform_dist(gen) * q);
            // 随机选择一个新的自旋状态。
            
            if (site >= 0 && site < L_x * L_y && new_spin >= 0 && new_spin < q) {
                double deltaE = calculateDeltaE(site, new_spin);
                // 计算能量变化。
                
                if (deltaE <= 0 || uniform_dist(gen) < std::exp(-beta * deltaE)) {
                    spins[site] = new_spin;
                    // 如果能量降低或满足Metropolis准则，则接受新的自旋状态。
                }
            }
        }
    }
};




int main() {
    // 系统参数设置
    int L_x = 2400; // 系统在x方向的尺寸
    int L_y = L_x; // 系统在y方向的尺寸，设置为与x方向相同
    int q = 3; // 自旋状态的数目
    double J = 1.0; // 相互作用强度
    double sigma_J = 0.0; // 相互作用的标准差
    double sigma_h = 0.00001; // 外场的标准差
    int seed = 12345; // 随机数生成器的种子
    std::string graph_type = "square";  // "hexagon"  // 图的类型


    // 模拟参数设置
    double beta_start = 0.1; // 起始的beta值
    double beta_end = 20.0; // 结束的beta值
    double beta_mult = 1.2; // beta的变化因子

    int warmup_steps = 1e2; // 热身步数
    int measure_interval = 1; // 测量间隔
    int n_states_record = 10; // 记录的状态数量
    int n_measures = 1e2; // 测量次数

    // 构建文件名
    std::string folder_path = "Potts_model2"; // 文件夹路径
    // 创建文件夹（如果不存在）
    try {
        if (!std::filesystem::exists(folder_path)) {
            std::filesystem::create_directory(folder_path); // 创建文件夹
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl; // 输出错误信息
        return 1; // 返回错误代码
    }

    // 构建文件名前缀
    std::string prefix = folder_path + "/Potts-"+ graph_type +"-"+ std::to_string(L_x) + "x" + std::to_string(L_y) +
                         "_q-" + std::to_string(q) + "_J-"+  std::to_string(J).substr(0, 4)+"_sJ-" + std::to_string(sigma_J).substr(0, 4) +
                         "_sh-" + std::to_string(sigma_h).substr(0, 6)+"_seed-" + std::to_string(seed);

    // 打印系统详细信息
    std::cout << "System Details:" << std::endl;
    std::cout << "L_x: " << L_x << ", L_y: " << L_y << ", q: " << q << std::endl;
    std::cout << "J: " << J << ", sigma_J: " << sigma_J << ", sigma_h: " << sigma_h << std::endl;
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Beta start: " << beta_start << ", Beta end: " << beta_end << ", Beta multiplier: " << beta_mult << std::endl;
    std::cout << "Warmup steps: " << warmup_steps << ", Measure interval: " << measure_interval << std::endl;
    std::cout << "Number of states to record: " << n_states_record << ", Number of measurements: " << n_measures << std::endl;
    std::cout << "Output prefix: " << prefix << std::endl;

    // 创建Potts模型对象
    PottsModel model(L_x, L_y, q, J, sigma_J, sigma_h, seed, graph_type);
    // 运行模拟
    model.runSimulation(beta_start, beta_end, beta_mult, 
                        warmup_steps, measure_interval, n_states_record, n_measures, prefix);

    return 0; // 程序成功结束
}




