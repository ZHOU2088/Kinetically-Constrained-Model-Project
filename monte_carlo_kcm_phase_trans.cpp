#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <utility>
#include <fstream>
#include <cmath>

// Define the possible movements for checking neighbors (up, down, left, right)
static const std::pair<int, int> directions[] = {
    {-1, 0}, // up
    {1, 0},  // down
    {0, -1}, // left
    {0, 1}    // right
};

const int SIZE = 60;
const int HalfSIZE = SIZE/2;
const double Zmin = 0.90;
const double Zmax = 0.901;
const double Zgap = 0.005;
const double Tmin = 0.99;
const double Tmax = 0.991; 
const double Tgap = 0.005;
const int NUM_INITIAL_ITER = 10;
const int R_MAX = HalfSIZE-1;
const long int NUM_ITERATIONS = 10000000;
const int NUM_RANDOM_INPUT = 1;


// Lambda function to check boundaries
bool inBounds(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

// Function to generate a random lattice configuration
void generateLattice(int** lattice, double Z) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Z stands for the probability of unoccupied sites
    std::bernoulli_distribution d(1 - Z);

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            lattice[i][j] = d(gen);
        }
    }
}

// Function to find blocked sites
void findBlockedSites(int** lattice, int** blocked) {
    
    if (lattice[0][0] == 1 || lattice[0][1] == 1 || lattice[1][0] == 1){
        blocked[0][0] = 1;
    }
    if (lattice[SIZE-1][0] == 1 || lattice[SIZE-1][1] == 1 || lattice[SIZE-2][0] == 1){
        blocked[SIZE-1][0] = 1;
    }
    if (lattice[0][SIZE-1] == 1 || lattice[0][SIZE-2] == 1 || lattice[1][SIZE-1] == 1){
        blocked[0][SIZE-1] = 1;
    }
    if (lattice[SIZE-1][SIZE-1] == 1 || lattice[SIZE-1][SIZE-2] == 1 || lattice[SIZE-2][SIZE-1] == 1){
        blocked[SIZE-1][SIZE-1] = 1;
    }
    
    for (int i = 1; i < SIZE-1; ++i) {
        if (lattice[i][0] == 1) {
            if (lattice[i - 1][0] == 1 || lattice[i + 1][0] == 1 || lattice[i][1] == 1){
                blocked[i][0] = 1;
            }
        }
        if (lattice[0][i] == 1) {
            if (lattice[0][i - 1] == 1 || lattice[0][i + 1] == 1 || lattice[1][i] == 1){
                blocked[0][i] = 1;
            }
        }
        
        if (lattice[i][SIZE-1] == 1) {
            if (lattice[i - 1][SIZE-1] == 1 || lattice[i + 1][SIZE-1] == 1 || lattice[i][SIZE-2] == 1){
                blocked[i][SIZE-1] = 1;
            }
        }
        if (lattice[SIZE-1][i] == 1) {
            if (lattice[SIZE-1][i - 1] == 1 || lattice[SIZE-1][i + 1] == 1 || lattice[SIZE-2][i] == 1){
                blocked[SIZE-1][i] = 1;
            }
        }
    }

    for (int i = 1; i < SIZE-1; ++i) {
        for (int j = 1; j < SIZE-1; ++j) {
            if (lattice[i][j] == 1) {
                if (lattice[i - 1][j] == 1 || lattice[i + 1][j] == 1 || 
                    lattice[i][j - 1] == 1 || lattice[i][j + 1] == 1) {
                    blocked[i][j] = 1;
                }
            }
        }
    }
}

// Refresh the blocked sites since we need count the nearest sites of the blocked sites 
void ReBlocksite(int** blocked) {

    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (blocked[i][j] == 1) {
                if (inBounds(i-1,j) && blocked[i-1][j] == 0) blocked[i-1][j] = 2;
                if (inBounds(i+1,j) && blocked[i+1][j] == 0) blocked[i+1][j] = 2;
                if (inBounds(i,j-1) && blocked[i][j-1] == 0) blocked[i][j-1] = 2;
                if (inBounds(i,j+1) && blocked[i][j+1] == 0) blocked[i][j+1] = 2;
            }
}

// Function to find clusters in the blocked sites
int DFS(int** blocked, bool** visited, int i, int j){
    int size = 0;
    std::vector<std::pair<int, int>> stack = {{i, j}};
    visited[i][j] = true;

    // Depth-First Search (DFS)
    while (!stack.empty()) {
        auto [x, y] = stack.back();
        stack.pop_back();
        size++;
    
    // Use static array for directional movement
        for (const auto& [dx, dy] : directions) {
            int nx = x + dx;
            int ny = y + dy;
            if (inBounds(nx, ny) && blocked[nx][ny] == 0 && !visited[nx][ny]) {
                visited[nx][ny] = true;
                stack.emplace_back(nx, ny);
            }
        }
    }
    return size;
}


int findLargestClusters(int** blocked) {
    bool** visited = new bool*[SIZE];
    bool out_of_maxclust = false;
    for(int i = 0; i < SIZE; i++){
        visited[i] = new bool[SIZE];
        std::fill(visited[i], visited[i] + SIZE, false);
    }
    int largest_clust = 0;
    int largest_i = 0;
    int largest_j = 0;

    // Traverse the blocked sites
    for (int i = 0; i < SIZE-1; ++i) {
        for (int j = 0; j < SIZE-1; ++j) {
            if (blocked[i][j] == 0 && !visited[i][j]) {
                int size = DFS(blocked, visited, i, j);
                if (size > largest_clust) {
                    largest_clust = size;
                    largest_i = i;
                    largest_j = j;
                    if (largest_clust > SIZE*SIZE/2) 
                        goto out_loop;
                }
            }
        }
    }
    out_loop:
    // refresh the visited array and do one DFS
    for (int i = 0; i < SIZE; ++i){
        for (int j = 0; j < SIZE; ++j)
        {
            visited[i][j] = false;
        }
    }
    int size = DFS(blocked, visited, largest_i, largest_j);
    if (!(size == largest_clust))
        std::cout << "the DFS code is wrong!!!" << std::endl;
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (visited[i][j]){
                blocked[i][j] = 3;  
            }    
    // Don't forget to free the memory allocated for visited
    for(int i = 0; i < SIZE; i++){
        delete[] visited[i];
    }
    delete[] visited;
    return largest_clust;
}

int hammingDistance(const std::vector<std::vector<int>>& config1, const std::vector<std::vector<int>>& config2) {
	int dist = 0;
	for (int i = 0; i < SIZE; ++i) {
	    for (int j = 0; j < SIZE; ++j) {
	        if (config1[i][j] != config2[i][j]) {
	           ++dist;
           }
	   }
    }
    return dist;
}
	


// Main function
int main() {
	// Allocate memory for the lattice and blocked arrays
	int** lattice = new int*[SIZE];
	for (int i = 0; i < SIZE; ++i) {
	   lattice[i] = new int[SIZE]();
	}
	int** blocked = new int*[SIZE];
	for (int i = 0; i < SIZE; ++i) {
	   blocked[i] = new int[SIZE]();
	}
	double Z = Zmin;
	// File operation
	std::string filename = "./output_dynamics="
	+ std::to_string(SIZE) + "_ITER=" + std::to_string(NUM_ITERATIONS) + "_Random-graph=" 
    + std::to_string(NUM_RANDOM_INPUT)+ "_INITIAL_ITER=" + std::to_string(NUM_INITIAL_ITER) + ".txt"; 
	std::ofstream outFile(filename);

	while (Z < Zmax) 
	{
	    // generate an infinite cluster and calculate the scale
        // Do it by Num_random_input times
        std::cout << "Z = "<< Z << std::endl;
        int scale = 0;
	    for (int iter = 0; iter < NUM_RANDOM_INPUT; ++iter) {
            std::cout << iter <<"nd random graph" << std::endl;
	        generateLattice(lattice, Z);
	       // The setup of the blocked array
	       for (int i = 0; i < SIZE; ++i){
	           for (int j = 0; j < SIZE; ++j){
	               blocked[i][j] = 0;
	           }
	        }
	        findBlockedSites(lattice, blocked);
	        ReBlocksite(blocked);
	        // Pick up the largest cluster
	        scale = findLargestClusters(blocked);
	    

	    // Monte Carlo for dynamics
	    for (double T = Tmin; T < Tmax; T += Tgap) {
            std::vector<std::vector<std::vector<int>>> canon_config;
            std::cout << "T = " << T << std::endl;
            for (int iteration_init = 0; iteration_init < NUM_INITIAL_ITER; iteration_init++){
                long int iteration = 0;
	            std::vector<std::vector<int>> config(SIZE, std::vector<int>(SIZE, 0));
	       
	            // Initialize configuration
                for (int i = 0; i < SIZE; ++i){
                    for (int j = 0; j < SIZE; ++j){
                        if (blocked[i][j] == 1) {
	                    // Set config[i][j] to 3 if blocked[i][j] == 1
	                    config[i][j] = 3;
	                    } else if (blocked[i][j] == 2) {
	                    // Set config[i][j] to 2 if blocked[i][j] == 2
	                    config[i][j] = 2;
	                    }
                    }
                }
                for (int k = 0; k < SIZE*SIZE; k++){
                    int i = rand() % SIZE;
	                int j = rand() % SIZE;
                    // Initialize config[i][j] based on blocked[i][j]
	                // Apply the probability rule for blocked[i][j] == 3
	                if (blocked[i][j] == 3) {
	                    config[i][j] = (rand() % 1000 < T * 1000) ? 1 : 0;
	                    // Check neighbors and change config[i][j] to 0 if necessary
	                    for (const auto& dir : directions) {
	                        int ni = i + dir.first;
	                        int nj = j + dir.second;
	                        if (inBounds(ni, nj) && config[ni][nj] == 1) {
	                            config[i][j] = 0;
	                            break; // No need to check further neighbors
	                        }
	                    }
	                }
	            }
            
	        // Monte Carlo dynamics
            while (iteration < NUM_ITERATIONS * scale / NUM_INITIAL_ITER){
                if (iteration + iteration_init * NUM_ITERATIONS * scale / NUM_INITIAL_ITER % (NUM_ITERATIONS * scale /5) == 0 ){
                     std::cout << "It is the "<< iteration + iteration_init * NUM_ITERATIONS * scale << "times" << std::endl;
                }
	            // Randomly choose a site
	            int i = rand() % SIZE;
	            int j = rand() % SIZE;
	            if (blocked[i][j] == 3) {
                    iteration++;
	                bool allNeighborsEmpty = true;
	                for (const auto& dir : directions) {
	                    int ni = i + dir.first;
	                       int nj = j + dir.second;
	                       if (inBounds(ni, nj) && config[ni][nj] == 1) {
	                           allNeighborsEmpty = false;
	                           break;
	                       }
	                }
	                if (allNeighborsEmpty) {
	                    config[i][j] = (rand() % 1000 < T * 1000) ? 1 : 0;
	                }
	            }
	        }
            // Calculate Hamming distance
	        bool is_unique = true; // Flag to check if the new configuration is unique
	        // Iterate over all configurations in canon_config
	        for (const auto& existing_config : canon_config) {
	        int dist = hammingDistance(config, existing_config);
	        if (dist <= static_cast<int>(std::ceil(scale * 0.3))) {
	            is_unique = false; // If the condition is met, the configuration is not unique
	            break; // No need to check further, we found a similar configuration
	        }
	        }
        	// If the configuration is unique, add it to canon_config
	        if (is_unique) {
	        canon_config.emplace_back(config);
        	}
            }
	        
	        // Output results
	        outFile << "Z = "<< Z << std::endl;
            outFile << "T = "<< T << std::endl;
            outFile << "It is the "<< iter << "nd graph with scale = " << scale << std::endl;           
            outFile << "Number of elements in canon_config: " << canon_config.size() << std::endl;  
	        for (const auto& c : canon_config) {
	            for (const auto& row : c) {
                    for (int val : row) {
	                    outFile << val << " ";
 	                }
	                outFile << std::endl;
	            }
                outFile << std::endl;
	        }
	        outFile << std::endl;
	    }
        }
	    Z += Zgap;
        
	}
	// Deallocate memory for the lattice and blocked arrays
	for (int i = 0; i < SIZE; ++i) {
	   delete[] lattice[i];
	   delete[] blocked[i];
	}
	delete[] lattice;
	delete[] blocked;    
	outFile.close();
	return 0;	
}
	
// Assume the existence of these functions

