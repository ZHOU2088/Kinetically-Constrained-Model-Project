#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <utility>
#include <fstream>The avarage energy is {E},
#include <cmath>

// Define the possible movements for checking neighbors (up, down, left, right)
static const std::pair<int, int> directions[] = {
    {-1, 0}, // upThe avarage energy is {E},
    {1, 0},  // down
    {0, -1}, // left
    {0, 1}    // rightThe avarage energy is {E},The avarage energy is {E},
};

const int SIZE = 100;

// parameters for initial states
const double Zmin = 0.9;
const double Zmax = 0.91;
const double Zgap = 0.5;
// parameters for temperature
const double Tmin = 20.99;
const double Tmax = 20.991; 
const double Tgap = 0.005;

// parameters for outfield variance and connection 
const double sigma = 2; 
const double J = -1;

// parameters for Monte Carlo simulation 
const int NUM_INITIAL_ITER = 5;
const long int NUM_ITERATIONS = 1000000;
const int NUM_RANDOM_INPUT = 5;


// function to check boundaries
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
            lattice[i][j] = round((d(gen)-0.5)*2);
        }
    }
}

// Function to generate a random outfield
void generateField(double** outField, double sigma){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, sigma);

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            outField[i][j] = d(gen);
        }
    }
}

void test_print(double** field){
    std::cout << "The field is" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < SIZE; ++i){
        for (int j = 0; j < SIZE; ++j){
            std::cout << field[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Main function
int main() {
	// Allocate memory for the lattice and blocked arrays
	int** lattice = new int*[SIZE];
	for (int i = 0; i < SIZE; ++i) {
	   lattice[i] = new int[SIZE]();
	}
    // Allocate memory for the lattice and blocked arrays 
	double** outField = new double*[SIZE];
	for (int i = 0; i < SIZE; ++i) {
	   outField[i] = new double[SIZE]();
	}
	double Z = Zmin;
	// File operation
	std::string filename = "./output_renormalized_SIZE="
	+ std::to_string(SIZE) + "_ITER=" + std::to_string(NUM_ITERATIONS) + "_Random-Field=" 
    + std::to_string(NUM_RANDOM_INPUT)+ "_INITIAL_ITER=" + std::to_string(NUM_INITIAL_ITER) + "_SIGMA=" + std::to_string(sigma) + ".txt"; 
	std::ofstream outFile(filename);

	while (Z < Zmax) 
	{
	    // generate a renormalized model
        // Do it by Num_random_input times
        // Initialize configuration

        std::cout << "Z = "<< Z << std::endl;

	    for (int iter = 0; iter < NUM_RANDOM_INPUT; ++iter) {
            std::cout << iter <<"nd random graph" << std::endl;
	        // The setup of outField
            generateField(outField, sigma);
            // test 
            test_print(outField);
	        // Monte Carlo for dynamics
	        for (double T = Tmin; T < Tmax; T += Tgap) {
                std::vector<std::vector<std::vector<int>>> canon_config;
                std::cout << "T = " << T << std::endl;
                for (int iteration_init = 0; iteration_init < NUM_INITIAL_ITER; iteration_init++){
                    long int iteration = 0;
                    // calculate the total energy 
                    double total_energy = 0; 
                    generateLattice(lattice, Z);
                    for (int i = 0; i < SIZE; ++i){
                        for (int j = 0; j < SIZE; ++j){
                            total_energy += lattice[i][j] * outField[i][j];
                            for (const auto& dir : directions) {
	                            int ni = i + dir.first;
	                            int nj = j + dir.second;
	                            if (inBounds(ni, nj)) {
                                total_energy += J*lattice[i][j] * lattice[ni][nj];
                                }
	                        }
                        }
                    }
	                // Monte Carlo dynamics
                    while (iteration < NUM_ITERATIONS * SIZE*SIZE / NUM_INITIAL_ITER){
                        if (iteration + iteration_init * NUM_ITERATIONS * SIZE*SIZE / NUM_INITIAL_ITER % (NUM_ITERATIONS * SIZE * SIZE /5) == 0 ){
                            std::cout << "It is the "<< iteration + iteration_init * NUM_ITERATIONS * SIZE * SIZE << "times" << std::endl;
                        }
	                    // Randomly choose a site
                        iteration++;
                        int i = rand() % SIZE;
	                    int j = rand() % SIZE;
	                    // Calculate the energy difference between the 
                        double exchange_energy = 0;
                        exchange_energy += -2*lattice[i][j] * outField[i][j];
	                    for (const auto& dir : directions) {
	                        int ni = i + dir.first;
	                        int nj = j + dir.second;
	                        if (inBounds(ni, nj)) {
                            exchange_energy += -2*J*lattice[i][j] * lattice[ni][nj];
                            }
	                    }
                        double prob = 1/(1+std::exp(T*exchange_energy)); 
                        if (rand() % 10000 < prob * 10000){
                            lattice[i][j] = -lattice[i][j];
                            total_energy += exchange_energy; 
                        }
	                }

                    // Output results
	                outFile << "Z = "<< Z << std::endl;
                    outFile << "T = "<< T << std::endl;
                    outFile << "E = "<< total_energy/(SIZE*SIZE) << std::endl;
                    outFile << "It is the "<< iter << "nd graph" << std::endl;
	                for (int i = 0; i < SIZE; i++) {
                        for (int j = 0; j < SIZE; j++) {
	                       outFile << lattice[i][j] << " ";
	                    }
                        outFile << std::endl;
	                }
	                outFile << std::endl;
                    // Next random_init 
	            }
                // Next temperature
	        }
            //Next random_field
	    }
        Z += Zgap;	
    }
	        

	// Deallocate memory for the lattice and blocked arrays
	for (int i = 0; i < SIZE; ++i) {
	   delete[] lattice[i];
	   delete[] outField[i];
	}
	delete[] lattice;
	delete[] outField;    
	outFile.close();
	return 0;	
}
	
// Assume the existence of these functions

