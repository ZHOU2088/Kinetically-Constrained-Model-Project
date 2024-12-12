#pragma GCC optimize(3,"Ofast","inline")
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <utility>
#include <thread>
#include <fstream>
#include <cmath>
#include <string>
#include <map>
#include <sstream>
#include <utility>
#include <chrono>

// Define the possible movements for checking neighbors (up, down, left, right)
static const std::pair<int, int> directions[] = {
    {-1, 0}, // up
    {1, 0},  // down
    {0, -1}, // left
    {0, 1}    // right
};

const int SIZE = 1000;
const int HalfSIZE = SIZE/2;
const double Zmin = 0.90;
const double Zmax = 0.901;
const double Zgap = 0.005;
const double Tmin = 0.0;
const double Tmax = 10.0; 
const double Tgap = 0.1;
const int NUM_INITIAL_ITER = 10;
const int R_MAX = HalfSIZE-1;
const long int NUM_ITERATIONS = 10000;
const int NUM_RANDOM_INPUT = 1;
const int NUM_PRERUN=5000;
const int NUM_SAMPLE = 5000;
const double P = 0.05;
const int K = 10;
 
std::pair<int,int>neighbor_index(int i,int j,int flag)
{
    std::pair<int,int>index;
    if(flag == 0)//上方
    {
        index = std::make_pair((i-1+SIZE)%SIZE,j);
    }
    else if(flag == 1)//右侧
    {
        index = std::make_pair(i,(j+1)%SIZE); 
    }
    else if(flag == 2)//下方
    {
        index = std::make_pair((i+1)%SIZE,j); 
    }
    else if(flag == 3)//左侧
    {
        index = std::make_pair(i,(j-1+SIZE)%SIZE); 
    }
    return index;
}
// Lambda function to check boundaries
bool inBounds(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

std::vector<std::pair<int,int>> loadConfiguration(int** lattice, std::ifstream& inputFile)
{
    
    std::string line;std::vector<std::pair<int,int>> free_spin;free_spin.reserve(SIZE*SIZE);
    int i = 0;int count =0;
    if (!inputFile.is_open()) 
    {
        std::cerr << "Unable to open file " << std::endl;
    }
    while(std::getline(inputFile,line) && i<SIZE)
    {
        if (line.empty()) 
        {
            std::cerr << "Warning: Empty line in configuration file at line " << i + 1 << std::endl;
            continue;
        }
        std::istringstream stream(line);
        int spin; int j =0;
        while (stream >> spin && j < SIZE) 
        {
            if(!spin)
            {
                free_spin.emplace_back(std::make_pair(i,j));
            }
            lattice[i][j] = (spin)? (-1):0;//blocked节点置-1，其余节点赋0
            j++;
        }
        i++;
    }
    return free_spin;
}

void performMCstep(int** lattice,double beta, std::vector<std::pair<int,int>>free_spin)
{
    std::random_device rd;
    std::mt19937 gen(rd());int size_FF = free_spin.size();
    std::uniform_int_distribution<int>ran(0,size_FF-1);
    std::uniform_real_distribution<double>d(0.0,1.0);
    for(int count=0;count<size_FF;count++)
    {
            // Randomly choose a free site
	            int index = ran(gen);
                int i = free_spin[index].first;
                int j = free_spin[index].second;
                if(inBounds(i,j))
                {
                    int new_spin = 1-lattice[i][j];
	                bool allNeighborsEmpty = true;
                    bool isConstrainUnsatisfied = false;
	                for (int flag =0; flag<4;flag++) 
                    {
	                    auto neighbor = neighbor_index(i,j,flag);
                        int ni = neighbor.first; int nj = neighbor.second;
	                    if (lattice[ni][nj] == 1 && new_spin == 1) 
                        {   
                            allNeighborsEmpty = false;
                            break;
	                    }
                        if(lattice[ni][nj] == 1 && new_spin == 0)
                        {
                            isConstrainUnsatisfied = true;
                            break;
                        }
	                }
	                if(isConstrainUnsatisfied)// flip when have 1-1 pair
                    {
                        lattice[i][j] = new_spin;
                    }
                    if (allNeighborsEmpty) //when the flip is legal 
                    {
	                    if(new_spin == 1 || d(gen) < exp(-1.0*beta))
                        {
                            lattice[i][j]=new_spin;
                        }
                        else//swap process for 1 spin that flip is rejected
                        {
                            int index_swap = ran(gen);
                            int swap_i = free_spin[index_swap].first;
                            int swap_j = free_spin[index_swap].second;
                            int swap_spin = lattice[swap_i][swap_j];
                            if(swap_spin != 1)
                            {
                                bool isNeighborsEmpty = true;
                                for (int flag =0; flag<4;flag++) 
                                {
	                                auto neighbor = neighbor_index(swap_i,swap_j,flag);
                                    int ni = neighbor.first; int nj = neighbor.second;
	                                if (lattice[ni][nj] == 1) 
                                    {   
                                        isNeighborsEmpty = false;
                                        break;
	                                }
	                            }
                                if(isNeighborsEmpty && d(gen) <0.5)
                                {
                                    lattice[i][j] = swap_spin;
                                    lattice[swap_i][swap_j] = 1;
                                }
                            }
                        }
	                }   
	                
                }
    }
	        
}

double getEnergyDensity(int** lattice)
{
    double e =0.0;
    double count =1.0;
    for (int i = 0; i < SIZE; ++i) 
    {
        for (int j = 0; j < SIZE; ++j) 
        {
            if(lattice[i][j]!=-1)
            {
                e+=lattice[i][j];
                count+=1.0;
            }
        }
        
    }
    return e/count;
}

void writeLatticeToFile(const std::string& filename, int** lattice) 
{
    std::ofstream file_out(filename);

    if (!file_out.is_open()) {
        std::cerr << "Unable to open output file: " << filename << std::endl;
        return;
    }
    file_out<<getEnergyDensity(lattice)<<std::endl;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            file_out << lattice[i][j] << " ";
        }
        file_out << "\n"; // New line for the next row
    }
    file_out.close();
}

int renormalize_spin(int i,int j,int**lattice,std::vector<std::pair<int,int>>&random_spin)
{
    int re_Lx = SIZE/K;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution d(0.5);
    int flag =0;
    for(int re_i = i*K;re_i<i*K+K;re_i++)
    {
        for(int re_j = j*K;re_j<j*K+K;re_j++)
        {
           if( lattice[re_i][re_j] == 1)//统计子晶格占据数
           { flag += (re_i+re_j)%2==0?1:-1;
           }
        }
    }
    if(flag == 0)//若区域内子晶格占据数相同，则随机分配一自旋方向
    {
        random_spin.emplace_back(std::make_pair(i,j));
        return 2*d(gen)-1;
        
    }
    else
    {
        return flag>0?1:-1;
    }
    
}

double renormalize_field(int**lattice,int re_i,int re_j)
{
    double h =0.0;
    for(int i =re_i*K;i<re_i*K+K;i++)
        {
            for(int j =re_j*K;j<re_j*K+K;j++)
            {
                if(lattice[i][j]!=-1)
                {
                    h+= (i+j)%2 ==0? 1.0:-1.0;
                }
            }
        }
    return h;
}

std::vector<double> renormalize_J(int**lattice,int re_i,int re_j)
{
    int re_Lx = SIZE/K; int re_Ly = re_Lx;
    std::vector<double>J_ij;J_ij.resize(4);
        //上方与下方
        for(int j = re_j*K;j<re_j*K+K;j++)
        {
            if(lattice[(re_i*K-1+SIZE)%SIZE][j]==0 && lattice[re_i*K][j]==0)
            {
                J_ij[0]+=1.0;
            }
            if(lattice[re_i*K+K-1][j]==0 && lattice[(re_i*K+K)%SIZE][j]==0)
            {
                J_ij[2]+=1.0;
            }
        }
        //左侧与右侧
        for(int i =re_i*K;i<re_i*K+K;i++ )
        {
            if(lattice[i][re_j*K]==0 && lattice[i][(re_j*K-1+SIZE)%SIZE]==0)
            {
                J_ij[3]+=1.0;
            }
            if(lattice[i][re_j*K+K-1]==0 && lattice[i][(re_j*K+K)%SIZE]==0)
            {
                J_ij[1]+=1.0;
            }
        }
        return J_ij;
    
}

int renormalize_neighbor(int re_i,int re_j,int flag)
{
     int Lx = SIZE/K; int Ly =  Lx;
    int neighbor_i =re_i;int neighbor_j =re_j;
        if(flag == 0)//上方
        {
            neighbor_i = (re_i-1+Lx)%Lx;
        }
        if(flag == 1)//右侧
        {
            neighbor_j = (re_j+1)%Ly;
        }
        if(flag == 2)//下方
        {
            neighbor_i = (re_i+1)%Lx;
        }
        if(flag == 3)//左侧
        {
            neighbor_j = (re_j-1+Ly)%Ly;
        }
        return neighbor_i*Lx+neighbor_j;
}

double renormalize_energy(std::vector<std::vector<int>>spin,std::vector<std::vector<double>> h,std::vector<std::vector<double>>J)
{
    int re_Lx = SIZE/K; int re_Ly = re_Lx;
    double energy =0.0;
    for(int i =0;i<re_Lx;i++)
    {
        for(int j =0;j<re_Ly;j++)
        {
           energy -= spin[i][j]*h[i][j];
        }    
    }
   
    for(int i =0;i<re_Lx;i++)
    {
        for(int j =0;j<re_Ly;j++)
        {
          for(int flag =0;flag <4;flag++)
          {
            int neighbor = renormalize_neighbor(i,j,flag);
            if(neighbor > i*re_Lx +j)
            {
                int n_i = neighbor/re_Lx; int n_j = neighbor%re_Lx;
                energy -= J[i*re_Lx+j][flag]*spin[i][j]*spin[n_i][n_j];
            }
          }
        }    
    }

    return energy;
}

double read_energy(std::vector<std::vector<int>>spin,std::string file_head)
{
    double energy =0.0;int re_Lx = SIZE/K;
    std::string file_name = file_head +"re_Ising.txt";
    std::ifstream file(file_name);
    if(!file.is_open())
    {
        throw std::runtime_error("Can not open file re_Ising.txt");
    }
    
    std::string line;
    int count =0;
    // 逐行读取文件
    while (std::getline(file, line) && count<re_Lx*re_Lx) 
    {
        int i = count/re_Lx; int j = count%re_Lx;
        std::istringstream iss(line);
        double number;
        if(iss>>number)//外场能量
        {
            energy -= spin[i][j]*number;
        }
        if(iss>>number)//上方邻居相互作用能量
        {
            energy-= number*0.5*spin[(i-1+re_Lx)%re_Lx][j]*spin[i][j];
        }
        if(iss>>number)//右侧邻居相互作用能量
        {
            energy-= number*0.5*spin[i][(j+1)%re_Lx]*spin[i][j];
        }
        if(iss>>number)//下方邻居相互作用能量
        {
            energy-= number*0.5*spin[(i+1)%re_Lx][j]*spin[i][j];
        }
        if(iss>>number)//左侧邻居相互作用能量
        {
            energy-= number*0.5*spin[i][(j-1+re_Lx)%re_Lx]*spin[i][j];
        }
        count++;
    }

    return energy;
}

void renormalize_lattice(std::vector<std::vector<double>>&field,std::vector<std::vector<double>>&J,std::string file_head,int**lattice)
{
    int re_Lx = SIZE/K; int re_Ly = re_Lx;
    field.reserve(re_Lx);
    J.reserve(re_Lx*re_Ly);
    for(int i =0;i<re_Lx;i++)
    {
        std::vector<double>h;h.reserve(re_Ly);
        for(int j =0;j<re_Ly;j++)
        {
            std::vector<double>J_ij;
           
            h.emplace_back(renormalize_field(lattice,i,j));
            J_ij = renormalize_J(lattice,i,j);
            J.emplace_back(J_ij);
        }
        field.emplace_back(h);
    }
    std::string file_name = file_head + "KCM_re.txt";
    std::ofstream file(file_name);
    if(!file.is_open())
    {
        throw std::runtime_error("Can't open file for output renormalize KCM lattice data.");
    }
    for(int i =0;i<re_Lx;i++)
    {
        for(int j =0;j<re_Ly;j++)
        {
            file<<field[i][j]<<" ";
            for(int flag =0;flag<4;flag++)
            {
                file<<J[i*re_Lx+j][flag]<<" ";
            }
            file<<std::endl;
        }
    }
    file.close();
}
void renormalize_process(int**lattice, std::string file_head,std::string out_re_name,std::vector<std::vector<double>>&h,std::vector<std::vector<double>>&J)
{
    int re_Lx = SIZE/K; int re_Ly = re_Lx;
    std::vector<std::vector<int>>spin;
    std::vector<std::pair<int,int>>random_spin;
    spin.reserve(re_Lx);random_spin.reserve(re_Lx*re_Ly);
    for(int i =0;i<re_Lx;i++)
    {
        std::vector<int>line;line.reserve(re_Ly);
       
        for(int j =0;j<re_Ly;j++)
        {
            
            line.emplace_back(renormalize_spin(i,j,lattice,random_spin));
        }
        spin.emplace_back(line);
    }
    double energy =0.0; double re_energy = 0.0;
    try
    {
        re_energy = read_energy(spin,file_head); 
        energy = renormalize_energy(spin,h,J); 
    } catch(const std::bad_alloc& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    }
    std::ofstream file(out_re_name);
    if(!file.is_open())
    {
        throw std::runtime_error("Can't open file for output renormalize KCM data.");
    }
    file<<"renormalize energy: "<<energy<<" energy read: "<<re_energy<<std::endl;
    if(random_spin.size()!=0)
    {
        for(int count =0;count<random_spin.size();count++)
        {
            file<<"( "<<random_spin[count].first<<" , "<<random_spin[count].second<<" ), ";
        }
        file<<std::endl;
    }
    for(int i =0;i<re_Lx;i++)
    {
        for(int j =0;j<re_Ly;j++)
        {
            file<<spin[i][j]<<" ";
        }
        file<<std::endl;
    }
    file.close();
}

void KCM_process(int count)
{
    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    int **lattice = new int *[SIZE];
        for (int i = 0; i < SIZE; ++i) 
        {
	    lattice[i] = new int[SIZE]();
	    }
        std::string folder ="./configuration/squareLattice/0.05/"+std::to_string(SIZE)+"/K_"+std::to_string(K);
        //load configuration in file
        std::string file_name = folder+"/"+std::to_string(count)+"/configuration.txt";
        std::string output_file = folder+"/"+std::to_string(count)+"/";
        std::string read_file = output_file+"test.txt";
        std::cout<<"thread "<<count<<" "<<file_name<<std::endl;
        std::ifstream File(file_name);
        if (!File.is_open()) 
        {
            std::cerr << "Unable to open file: " << file_name << std::endl;
        
        }
        auto free_spin = loadConfiguration(lattice,File);
        File.close();
        std::vector<std::vector<double>>field;std::vector<std::vector<double>>J;
        renormalize_lattice(field,J,output_file,lattice);

        // Monte Carlo for dynamics
	    for (double T = Tmin; T <= Tmax; T += Tgap) 
        {
            //prerun
            std::cout<<"thread "<<count<<" beta:"<<T<<std::endl;
            for(int flag =0;flag<NUM_PRERUN;flag++)
            {
                performMCstep(lattice,T,free_spin);
            }
            //MC dynamics
            for(int iter = 0; iter<NUM_ITERATIONS;iter++)
            {
                performMCstep(lattice,T,free_spin);
            }
            std::cout<<"thread "<<count<<" energy density: "<<getEnergyDensity(lattice)<<std::endl;
            
        }

        for(int i = 0;i<10;i++)
        {
            for(int j =0;j<NUM_PRERUN;j++)
            {
                performMCstep(lattice,Tmax,free_spin);

            }
            std::string out_name = output_file+"end_KCM_"+std::to_string(i)+".txt";
            std::string out_re_name = output_file+"end_KCM_re_"+std::to_string(i)+".txt";
            writeLatticeToFile(out_name,lattice);
            //renormalize_process
           try
           {
                renormalize_process(lattice,output_file,out_re_name,field,J); 
           }catch(const std::bad_alloc& e)
           {
                std::cerr << "Memory allocation failed: " << e.what() << std::endl;
           }

        }
        // Deallocate memory for the lattice 
	    for (int i = 0; i < SIZE; ++i) 
        {
	    delete[] lattice[i];
	    }
	    delete[] lattice;  

        // 结束计时并输出执行时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
	std::cout<<"thread "<<count<<" finish calculation! Time use: "<<elapsed_seconds.count()<<" seconds"<<std::endl;
}
// Main function
int main() 
{
    int numParts = 10; // 假设操作被分解为10个独立部分
    
    // 使用 vector 存储线程对象
    
    std::vector<std::thread> threads;

    // 为每个操作部分创建并启动一个线程
    for (int i = 0; i < numParts; ++i) {
        threads.emplace_back(KCM_process, i);
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    std::cout << "All tasks processed." << std::endl;

    return 0;
}
    
// Assume the existence of these functions