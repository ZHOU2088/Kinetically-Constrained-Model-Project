#pragma GCC optimize(3,"Ofast","inline")
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



// g++ -O3 -fopenmp -march=native MC_renormalize_Ising.cpp -o KCM
// ./CL


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

    // 按照 remain_flag 过滤 state 数组
    std::vector<int> filter_state() {
        std::vector<int> filtered_state;
        for (size_t i = 0; i < states.size(); ++i) {
            if (remain_flag[i] == 1) {
                filtered_state.push_back(0);
            }
            else
            {
                filtered_state.push_back(1);
            }
        }
        return filtered_state;
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

        states = filter_state();

        return new_neighbors_list; 
    }

    double renormalize_field(int re_i,int re_j,int k)
    {
        double h = 0.0;
        for(int i =re_i*k;i<re_i*k+k;i++)
        {
            for(int j =re_j*k;j<re_j*k+k;j++)
            {
                int index = i*L_x+j;
                if(states[index]==0)
                {
                    h+= (i+j)%2 ==0? 1.0:-1.0;
                }
            }
        }
        return h;
    }
    
    std::vector<double> renormalize_J(int re_i,int re_j,int k)
    {
        std::vector<double>J_ij;J_ij.resize(4);
        int i_up = (re_i*k-1+L_x)%L_x; int i_down =(re_i*k+k)%L_x;
        int j_right = (re_j*k+k)%L_y;int j_left = (re_j*k-1+L_y)%L_y;
        
        //上方与下方
        for(int j = re_j*k;j<re_j*k+k;j++)
        {
            //std::cout<<i_down*L_x+j<<" "<<states[i_down*L_x+j];
            //std::cout<<" "<<(re_i*k+k-1)*L_x+j<<" "<<states[(re_i*k+k-1)*L_x+j]<<std::endl;
            if(states[i_up*L_x+j]==0 && states[(re_i*k)*L_x+j]==0)
            {
                J_ij[0]+=1.0;
            }
            if(states[i_down*L_x+j]==0 && states[(re_i*k+k-1)*L_x+j]==0)
            {
                J_ij[2]+=1.0;
            }
        }
        //左侧与右侧
        for(int i =re_i*k;i<re_i*k+k;i++ )
        {
            if(states[i*L_x+j_left]==0 && states[i*L_x+re_j*k]==0)
            {
                J_ij[3]+=1.0;
            }
            if(states[i*L_x+re_j*k+k-1]==0 && states[i*L_x+j_right]==0)
            {
                J_ij[1]+=1.0;
            }
        }
        return J_ij;
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

void save_configuration(std::ofstream& ofs, LatticeProcessor processor) {
    if (!ofs.is_open()) {
        throw std::runtime_error("File stream is not open for writing.");
    }
    for( int i = 0; i < processor.L_x; i++)
    {
        for(int j = 0; j< processor.L_y;j++)
        {
            int index = i*processor.L_x+j;
            ofs<<processor.states[index]<<" ";
        }
        ofs<<std::endl;
    }
}

class re_lattice
{
    private:
    int K;int seed;
    int Lx;int Ly;int N;
    double energy;
    std::vector<int>states;
    std::vector<double>field;
    std::vector<std::vector<double>>J;
    public:
    re_lattice(LatticeProcessor lattice,std::string file_head,int k,int seed)
    {
        K = k;  Lx = lattice.L_x/K; Ly = lattice.L_y/K; N = Lx*Ly;
        states.reserve(N);field.reserve(N);J.reserve(N);
        std::mt19937 gen(seed); 
        std::bernoulli_distribution d(0.5);
        //打开文件
        std::string output_re = file_head + "/re_Ising.txt";
        std::ofstream file_re(output_re);
        if(!file_re.is_open())
        {
            throw std::runtime_error("File stream is not open for writing.");
        }
        for(int i =0;i<Lx;i++)
        {
            for(int j =0;j<Ly;j++)
            {
                //计算重整化外场
                double h = lattice.renormalize_field(i,j,K);
                field.emplace_back(h);
                file_re<<h<<" ";
                //计算重整化相互作用
                auto J_ij = lattice.renormalize_J(i,j,K);
                file_re<<J_ij[0]<<" "<<J_ij[1]<<" "<<J_ij[2]<<" "<<J_ij[3]<<" "<<std::endl;
                J.emplace_back(J_ij);
                //初始化晶格
                states.emplace_back(2*d(gen)-1);
            }
        }
        //关闭文件
        file_re.close();
        energy =0.0;
    }
    int getIndex(int i,int j)
    {
        return i*Lx+j;
    }
    int getNeighborIndex(int i,int j,int flag)
    {
        int neighbor_i =i;int neighbor_j =j;
        if(flag == 0)//上方
        {
            neighbor_i = (i-1+Lx)%Lx;
        }
        if(flag == 1)//右侧
        {
            neighbor_j = (j+1)%Ly;
        }
        if(flag == 2)//下方
        {
            neighbor_i = (i+1)%Lx;
        }
        if(flag == 3)//左侧
        {
            neighbor_j = (j-1+Ly)%Ly;
        }
        return neighbor_i*Lx+neighbor_j;
    }
    double calculateEnergy()
    {
        double energy = 0.0;
        for(int i =0;i<Lx;i++)
        {
            for(int j =0; j<Ly;j++)
            {
                //计算外场能量
                int index = i*Lx+j;
                energy -= field[index]*states[index];
                //计算相互作用能
                for(int flag =0;flag<4;flag++)
                {
                    int neighbor = getNeighborIndex(i,j,flag);
                    if(neighbor > index)
                    {
                        energy -= J[index][flag]*states[neighbor]*states[index];
                    }
                }
            }
        }
        return energy;
    }
    double DeltaE(int i,int j)
    {
        
        int index = i*Lx+j; int spin = states[index];
        double deltaE = 0.0; deltaE += 2*field[index]*spin;
        for(int flag = 0;flag<4;flag++)
        {
            int neighbor = getNeighborIndex(i,j,flag);
            deltaE += 2*J[index][flag]*spin*states[neighbor];
        }
        return deltaE;

        
    }
    void performMCstep(double beta,int seed)
    {
        std::mt19937 gen(seed); 
        std::uniform_real_distribution<double>ran(0.0,1.0);
        std::uniform_int_distribution<int>random_i(0,Lx-1);
        std::uniform_int_distribution<int>random_j(0,Ly-1);
        for(int count=0;count<N;count++)
        {
            int i = random_i(gen); int j = random_j(gen);
            double deltaE = DeltaE(i,j);
            if(deltaE <=0 || ran(gen) < exp(-1.0*beta*deltaE))
            {
                states[i*Lx+j] *= -1; 
            }
        }
    }
    void MCdynamics(double beta_start,double beta_end,double beta_step,int pre_step,int num_iter,int sample_interval,std::string file_head)
    {
        
        for(double beta = beta_start;beta<beta_end;beta+=beta_step)
        {
             double e = 0.0;double e_var =0.0;
            for(int count =0;count < pre_step;count++)
            {
                performMCstep(beta,seed);
            }
            int i =0;
            for(int count = 0; count<num_iter;count++)
            {
                performMCstep(beta,seed);
                if(count%sample_interval == 0)
                {
                    double e_i = calculateEnergy();
                    e+=e_i;e_var+=e_i*e_i;i++;
                }
            }
            e/=i;e_var/=i;
            std::cout<<beta<<" "<<e<<" "<<e_var-e*e<<std::endl;
           
            
        }
        for(int i =0;i<10;i++)
        {
            std::string file_end_name = file_head + "/end_Ising_"+std::to_string(i)+".txt";
            std::ofstream file_end(file_end_name);
            if(!file_end.is_open())
            {
                throw std::runtime_error("Could't open file_end!");
            }
            for(int count =0;count < pre_step;count++)
            {
                performMCstep(beta_end,seed);
            }
            file_end<<"energy: "<<calculateEnergy()<<std::endl;
            for(int c =0; c<Lx;c++)
            {
                for(int l =0;l<Ly;l++)
                {
                    file_end<<states[c*Lx+l]<<" ";
                }
                file_end<<std::endl;
            }
            file_end.close();
        }
    }
};


int main() 
{
    // 初始化固定参数
    int graph_seed = 1919810;
    int threshold = 1;
    int num_samples = 10;
    int K = 20;
    std::string graph_type = "square";
    std::string type = "renormalize";//MC_renormalize or renormalize

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 遍历不同的 L_x 和 p_0
    std::vector<int> L_x_values = {1000};  
    std::vector<double> p_0_values = {0.20};  

    for (int L_x : L_x_values) {
        int L_y = L_x;  // 假设 L_y 与 L_x 相同
        for (double p_0 : p_0_values) {
            std::cout << "L_x = " << L_x << ", p_0 = " << p_0 <<", K = "<<K<< std::endl;
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
            
            std::string RandomGraphType = graph_type+"Lattice";
            std::string folder = "./configuration/"+ RandomGraphType;
            std::filesystem::create_directories(folder);
            std::string folder_P = folder+"/"+p0_str+"/"+std::to_string(L_x)+"/K_"+std::to_string(K);
            std::filesystem::create_directories(folder_P);
            for (int i = 0; i < num_samples; ++i) 
            {
                //文件操作
                std::cout<<"i: "<<i<<std::endl;
                LatticeProcessor processor(L_x, L_y, threshold, p_0, graph_seed + i,graph_type);
                processor.process();
                std::string folder_i = folder_P+"/"+std::to_string(i);
                std::filesystem::create_directories(folder_i);
                std::string fileName = folder_i+"/configuration.txt";
                std::string re_fileName = folder_i+"/re_Ising.txt"; 
                std::ofstream file(fileName);
                if (!file.is_open()) 
                {
                throw std::runtime_error("File stream is not open for writing.");
                }
                save_configuration(file,processor);
                file.close();
                //重整化过程
                re_lattice Ising(processor,folder_i,K,graph_seed);
                Ising.MCdynamics(0.0,5.0,0.01,5000,50000,5000,folder_i);
            }
            
        }
    }

    // 结束计时并输出执行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "Total time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}