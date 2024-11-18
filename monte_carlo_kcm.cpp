#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <utility>
#include <fstream>

// Define the possible movements for checking neighbors (up, down, left, right)
static const std::pair<int, int> directions[] = {
    {-1, 0}, // up
    {1, 0},  // down
    {0, -1}, // left
    {0, 1}    // right
};

const int SIZE = 3000;
const int HalfSIZE = SIZE/2;
const double Z = 0.547;
const int R_MAX = HalfSIZE-1;
const int NUM_ITERATIONS = 50;
const int Max_clust = 1000;
const int Max_area_clust = 2000;

// Lambda function to check boundaries
bool inBounds(int x, int y) {
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

// Function to generate a random lattice configuration
void generateLattice(int** lattice) {
    std::random_device rd;
    std::mt19937 gen(rd());
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

// Function to count blocked sites along the boundary of the square
void countBoundarySites(int** lattice, int** blocked, int r, int* numBlocked, int* numZero) {
    *numBlocked = 0;
    *numZero = 0;

    for (int i = HalfSIZE - r + 1; i <= HalfSIZE + r - 1; ++i) {
        if (blocked[i][HalfSIZE - r] == 1)
            (*numBlocked)++;
        if (lattice[i][HalfSIZE - r] == 0)
            (*numZero)++;
        
        if (blocked[i][HalfSIZE + r] == 1)
            (*numBlocked)++;
        if (lattice[i][HalfSIZE + r] == 0)
            (*numZero)++;
        
        if (blocked[HalfSIZE - r][i] == 1)
            (*numBlocked)++;
        if (lattice[HalfSIZE - r][i] == 0)
            (*numZero)++;
        
        if (blocked[HalfSIZE + r][i] == 1)
            (*numBlocked)++;
        if (lattice[HalfSIZE + r][i] == 0)
            (*numZero)++;
      
    }

    if (blocked[HalfSIZE - r][HalfSIZE - r] == 1)
        (*numBlocked)++;
    if (lattice[HalfSIZE - r][HalfSIZE - r] == 0)
        (*numZero)++;
    
    if (blocked[HalfSIZE + r][HalfSIZE - r] == 1)
        (*numBlocked)++;
    if (lattice[HalfSIZE + r][HalfSIZE - r] == 0)
        (*numZero)++;
    
    if (blocked[HalfSIZE - r][HalfSIZE + r] == 1)
        (*numBlocked)++;
    if (lattice[HalfSIZE - r][HalfSIZE + r] == 0)
        (*numZero)++;
        
    if (blocked[HalfSIZE + r][HalfSIZE + r] == 1)
        (*numBlocked)++;
    if (lattice[HalfSIZE + r][HalfSIZE + r] == 0)
        (*numZero)++;       
}
// Refresh the blocked sites since we need count the nearest sites of the blocked sites 
void ReBlocksite(int** blocked) {

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            if (blocked[i][j] == 1) {
                if (inBounds(i-1,j) && blocked[i-1][j] == 0) blocked[i-1][j] = 2;
                if (inBounds(i+1,j) && blocked[i+1][j] == 0) blocked[i+1][j] = 2;
                if (inBounds(i,j-1) && blocked[i][j-1] == 0) blocked[i][j-1] = 2;
                if (inBounds(i,j+1) && blocked[i][j+1] == 0) blocked[i][j+1] = 2;
            }

    /*std::cout << "The reblocked one is" << std::endl;
    for (int i = 0; i < 20; ++i){
        for (int j = 0; j < 20; ++j){
            std::cout << blocked[i][j];
        }
        std::cout << std::endl;
    }*/
}

// Function to find clusters in the blocked sites


void findClusters(int** blocked, int* clusterSizes, int* numClusters, int* clusterNumBySize) {
    bool** visited = new bool*[SIZE];
    bool out_of_maxclust = false;
    for(int i = 0; i < SIZE; i++){
        visited[i] = new bool[SIZE];
        std::fill(visited[i], visited[i] + SIZE, false);
    }
    *numClusters = 0;
    std::cout << "cluster sizes:"; 
    int largest_clust = 0;

    // Traverse the blocked sites
    for (int i = 0; i < SIZE-1; ++i) {
        for (int j = 0; j < SIZE-1; ++j) {
            if (blocked[i][j] == 0 && !visited[i][j]) {
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
                if (size > 0) {
                    if (*numClusters < Max_clust) clusterSizes[*numClusters] = size;
                    if (size > largest_clust) largest_clust = size;
                    (*numClusters)++;
                    if (*numClusters < Max_clust) std::cout << size << "  ";
                    if (size < Max_area_clust) clusterNumBySize[size]++;
                }
                if (*numClusters >= Max_clust && !out_of_maxclust){
                    std::cout << "OUT OF DEFAULT NUMBER OF CLUSTERS!";
                    out_of_maxclust = true;
                }
            }
        }
    }
    std::cout << std::endl;
    // Don't forget to free the memory allocated for visited
    for(int i = 0; i < SIZE; i++){
        delete[] visited[i];
    }
    delete[] visited;
    int sort_num = *numClusters;
    if (out_of_maxclust) sort_num = Max_clust; 
    std::sort(clusterSizes, clusterSizes + sort_num, std::greater<int>());
    clusterSizes[0] = largest_clust;
    std::cout << "the area of the largest cluster: " << largest_clust << std::endl;
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

    // Rows for blocked and zero counts and the clusters' counts and scales 
    int clusterSizes[Max_clust] = {0};
    int clusterNumBySize[Max_area_clust+1] = {0};
    long long avgclustSizes[Max_clust] = {0};
    double avgclusterNumBySize[Max_area_clust+1] = {0};
    double avgBlocked[R_MAX] = {0};
    double avgZero[R_MAX] = {0};
    double avgClusters = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        generateLattice(lattice);
        //The setup of the blocked array 
        for (int i = 0; i < SIZE; ++i){
            for (int j = 0; j < SIZE; ++j){
                blocked[i][j] = 0;
            }
        }
        findBlockedSites(lattice, blocked);
        /*
        for (int r = 1; r <= R_MAX; ++r) {
            int numBlocked = 0, numZero = 0;
            countBoundarySites(lattice, blocked, r, &numBlocked, &numZero);
            avgBlocked[r - 1] += numBlocked;
            avgZero[r - 1] += numZero;
        }
        */

        int numClusters = 0;
        ReBlocksite(blocked);

        // initialized clustNumbysize
        for (int i = 1; i <= Max_area_clust; i++){
            clusterNumBySize[i] = 0; 
        }
        findClusters(blocked, clusterSizes, &numClusters, clusterNumBySize);
        
        avgClusters += numClusters;
        // Transfer to the ensemble average 
        for(int i = 0; i < Max_clust; ++i){
            avgclustSizes[i] += clusterSizes[i];
        }
        for(int i = 1; i <= Max_area_clust; ++i){
            avgclusterNumBySize[i] += clusterNumBySize[i];
        }
        std::cout<< "current iteration:" << iter << std::endl; 
    }
    // test of configuration
        for (int i = 0; i < 20; ++i){
        for (int j = 0; j < 20; ++j){
            std::cout << blocked[i][j];
        }
        std::cout << std::endl;
        }
    // Calculate averages
    /*
    for (int i = 0; i < R_MAX; ++i){
        avgBlocked[i] /= (NUM_ITERATIONS);
        avgZero[i] /= (NUM_ITERATIONS);
    }
    */
    avgClusters /= NUM_ITERATIONS;

    for(int i = 0; i < Max_clust; i++){
        avgclustSizes[i] /= NUM_ITERATIONS;
    }

    for(int i = 1; i <= Max_area_clust; i++){
        avgclusterNumBySize[i] /= NUM_ITERATIONS;
    }

    // File operation
    std::string filename = "./output_KCM_Z=" + std::to_string(Z) + "_SIZE=" 
    + std::to_string(SIZE) + "_ITER=" + std::to_string(NUM_ITERATIONS) + ".txt"; 
    std::ofstream outFile(filename);

    // Display the averages
    outFile << std::fixed << std::setprecision(5);

    /*
    outFile << "Average number of blocked sites: " << std::endl;
    for (int i = 0; i < R_MAX; ++i){
        outFile << avgBlocked[i] << " ";  
    }
    outFile << std::endl;
    outFile << "Average proportion of blocked sites: " << std::endl;
    for (int i = 0; i < R_MAX; ++i){
        outFile << avgBlocked[i] / (8 * i + 8) << " ";  
    }
    outFile << std::endl;
    outFile << "Average number of zero sites: " << std::endl;
    for (int i = 0; i< R_MAX; ++i){
        outFile << avgZero[i] << " ";
    }
    outFile << std::endl;
    outFile << "Average proportion of zero sites: " << std::endl;
    for (int i = 0; i< R_MAX; ++i){
        outFile << avgZero[i] / (8 * i + 8) << " ";
    }
    outFile << std::endl;
    */

    outFile << "Average number of clusters: " << avgClusters << std::endl;
    outFile << "Average number of scales: " << std::endl;
    for(int i = 0; i < Max_clust; i++){
        outFile << avgclustSizes[i] << " ";
    }
    outFile << std::endl;
    outFile << "Avarage number of clusters by size: ";
    for(int i = 1; i <= Max_area_clust; i++){
        outFile << avgclusterNumBySize[i] << " ";
    }
    outFile.close();
    // Deallocate memory for the lattice and blocked arrays

    for (int i = 0; i < SIZE; ++i) {
        delete[] lattice[i];
        delete[] blocked[i];
    }
    delete[] lattice;
    delete[] blocked;

    return 0;
}