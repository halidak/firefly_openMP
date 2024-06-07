#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>  
#include <cstdlib>  
#include <random>

using namespace std;
using namespace std::chrono;

//-----------Unimodalne funkcije----------------
// Ciljna funkcija sum squares
double sumOfSquaresFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

// Benchmark funkcija Step2
double step2Function(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += pow(floor(xi) + 0.5, 2);
    }
    return sum;
}

// Benchmark funkcija Quartic
double quarticFunction(const vector<double>& x) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (i + 1) * pow(x[i], 4) + dist(gen);
    }
    return sum;
}

// Powellova funkcija
double powellFunction(const vector<double>& x) {
    double sum = 0.0;
    int n = x.size();
    for (int i = 0; i < n / 4; ++i) {
        double term1 = x[4 * i] + 10 * x[4 * i + 1];
        double term2 = x[4 * i + 2] - x[4 * i + 3];
        double term3 = x[4 * i + 1] - 2 * x[4 * i + 2];
        double term4 = x[4 * i] - x[4 * i + 3];
        sum += term1 * term1 + 5 * term2 * term2 + term3 * term3 * term3 * term3 + 10 * term4 * term4 * term4 * term4;
    }
    return sum;
}

// Benchmark funkcija Rosenbrock
double rosenbrockFunction(const vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size() - 1; ++i) {
        sum += 100 * pow(x[i + 1] - x[i] * x[i], 2) + pow(x[i] - 1, 2);
    }
    return sum;
}

// Benchmark funkcija Dixon-Price
double dixonPriceFunction(const vector<double>& x) {
    double sum = pow(x[0] - 1, 2);
    for (size_t i = 1; i < x.size(); ++i) {
        sum += i * pow(2 * x[i] * x[i] - x[i - 1], 2);
    }
    return sum;
}

// Benchmark funkcija Schwefel 1.2
double schwefel12Function(const vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double innerSum = 0.0;
        for (size_t j = 0; j <= i; ++j) {
            innerSum += x[j];
        }
        sum += innerSum * innerSum;
    }
    return sum;
}

// Benchmark funkcija Schwefel 2.20
double schwefel220Function(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += abs(xi);
    }
    return sum;
}

// Benchmark funkcija Schwefel 2.21
double schwefel221Function(const vector<double>& x) {
    double maxVal = 0.0;
    for (double xi : x) {
        maxVal = max(maxVal, abs(xi));
    }
    return maxVal;
}

// Sphere Function
double sphereFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

//-----------Multimodalne funkcije----------------
// Benchmark funkcija Rastrigin
double rastriginFunction(const vector<double>& x) {
    int d = x.size();
    double sum = 10 * d;
    for (double xi : x) {
        sum += (xi * xi - 10 * cos(2 * M_PI * xi));
    }
    return sum;
}

// Benchmark funkcija Griewank
double griewankFunction(const vector<double>& x) {
    double sum1 = 0.0;
    double sum2 = 1.0;
    int d = x.size();

    for (int i = 0; i < d; ++i) {
        sum1 += x[i] * x[i];
        sum2 *= cos(x[i] / sqrt(i + 1));
    }

    return 1.0 + sum1 / 4000.0 - sum2;
}

// Csendes Function
double csendesFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += pow(xi, 6) * (2 + sin(1.0 / xi));
    }
    return sum;
}

// Colville Function
double colvilleFunction(const vector<double>& x) {
    return 100 * pow(x[1] - pow(x[0], 2), 2) +
        pow(x[0] - 1, 2) +
        pow(x[2] - 1, 2) +
        90 * pow(x[3] - pow(x[2], 2), 2) +
        10.1 * (pow(x[1] - 1, 2) + pow(x[3] - 1, 2)) +
        19.8 * (x[1] - 1) * (x[3] - 1);
}

// Easom Function
double easomFunction(const vector<double>& x) {
    return -cos(x[0]) * cos(x[1]) * exp(-pow(x[0] - M_PI, 2) - pow(x[1] - M_PI, 2));
}

// Michalewicz Function
double michalewiczFunction(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += sin(x[i]) * pow(sin((i + 1) * x[i] * x[i] / M_PI), 20);
    }
    return -sum;
}

// Shekel Function
double shekelFunction(const vector<double>& x) {
    const int m = 10;
    const double a[m][4] = {
        {4.0, 4.0, 4.0, 4.0}, {1.0, 1.0, 1.0, 1.0}, {8.0, 8.0, 8.0, 8.0},
        {6.0, 6.0, 6.0, 6.0}, {3.0, 7.0, 3.0, 7.0}, {2.0, 9.0, 2.0, 9.0},
        {5.0, 5.0, 3.0, 3.0}, {8.0, 1.0, 8.0, 1.0}, {6.0, 2.0, 6.0, 2.0},
        {7.0, 3.6, 7.0, 3.6}
    };
    const double c[m] = { 0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5 };

    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double inner_sum = 0.0;
        for (int j = 0; j < 4; ++j) {
            inner_sum += pow(x[j] - a[i][j], 2);
        }
        sum += 1.0 / (inner_sum + c[i]);
    }
    return -sum;
}

// Schwefel 2.4 Function
double schwefel24Function(const vector<double>& x) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += pow(x[i] - 1, 2) + (i > 0 ? pow(x[i] - x[i - 1] * x[i - 1], 2) : 0);
    }
    return sum;
}

// Schwefel Function
double schwefelFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += -xi * sin(sqrt(abs(xi)));
    }
    return 418.9829 * x.size() - sum;
}

// Schaffer Function
double schafferFunction(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size() - 1; ++i) {
        double temp1 = x[i] * x[i] + x[i + 1] * x[i + 1];
        double temp2 = sin(sqrt(temp1)) * sin(sqrt(temp1)) - 0.5;
        sum += 0.5 + temp2 / pow(1 + 0.001 * temp1, 2);
    }
    return sum;
}

// Alpine Function
double alpineFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += abs(xi * sin(xi) + 0.1 * xi);
    }
    return sum;
}

// Ackley Function
double ackleyFunction(const vector<double>& x) {
    int d = x.size();
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < d; ++i) {
        sum1 += x[i] * x[i];
        sum2 += cos(2 * M_PI * x[i]);
    }

    return -20.0 * exp(-0.2 * sqrt(sum1 / d)) - exp(sum2 / d) + 20.0 + M_E;
}

// Ciljna funkcija Schwefel 2.22
double schwefel222Function(const vector<double>& x) {
    double sum = 0.0;
    double prod = 1.0;

    for (double xi : x) {
        sum += abs(xi);
        prod *= abs(xi);
    }

    return sum + prod;
}

typedef double (*BenchmarkFunction)(const vector<double>&);

void fireflyAlgorithm(int n, int d, int maxGenerations, int numThreads, BenchmarkFunction benchmarkFunction, vector<double>& results, vector<double>& meanBestPerGeneration) {
    const double alpha = 0.5; // Initial value of alpha
    const double betaMin = 0.2;
    const double gamma = 1.0;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    // Initialization of fireflies
    vector<vector<double>> fireflies(n, vector<double>(d));
    vector<double> lightIntensity(n);

    // Parallel initialization
#pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            fireflies[i][j] = dist(gen);
        }
        lightIntensity[i] = benchmarkFunction(fireflies[i]);
    }

    for (int t = 0; t < maxGenerations; ++t) {
        // Parallel main loop
#pragma omp parallel for num_threads(numThreads)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (lightIntensity[j] < lightIntensity[i]) {
                    double r = 0.0;
                    for (int k = 0; k < d; ++k) {
                        r += pow(fireflies[i][k] - fireflies[j][k], 2);
                    }
                    r = sqrt(r);

                    double beta = betaMin + (1.0 - betaMin) * exp(-gamma * r * r);

                    for (int k = 0; k < d; ++k) {
                        double randVal = rand01(gen);
                        fireflies[i][k] = fireflies[i][k] * (1.0 - beta) + fireflies[j][k] * beta + alpha * (randVal - 0.5);
                    }
                    lightIntensity[i] = benchmarkFunction(fireflies[i]);
                }
            }
        }

        double bestLightIntensity = *min_element(lightIntensity.begin(), lightIntensity.end());
        meanBestPerGeneration.push_back(bestLightIntensity);
    }

    double globalBest = *min_element(lightIntensity.begin(), lightIntensity.end());
    results.push_back(globalBest);
}

int main() {
    srand(time(0));  // Initialize random seed

    int n = 50;  // Number of fireflies
    int d = 30;  // Problem dimension
    int maxGenerations = 512;  // Maximum number of generations
    int numThreads = 2;  // Number of threads

    // Define an array of benchmark functions
    BenchmarkFunction benchmarkFunctions[] = {
        schwefel222Function,
        sumOfSquaresFunction,
        step2Function,
        quarticFunction,
        powellFunction,
        rosenbrockFunction,
        dixonPriceFunction,
        schwefel12Function,
        schwefel220Function,
        schwefel221Function,
        sphereFunction,
        rastriginFunction,
        griewankFunction,
        csendesFunction,
        colvilleFunction,
        easomFunction,
        michalewiczFunction,
        shekelFunction,
        schwefel24Function,
        schwefelFunction,
        schafferFunction,
        alpineFunction,
        ackleyFunction,
    };

    int numFunctions = sizeof(benchmarkFunctions) / sizeof(benchmarkFunctions[0]);

    // Loop over each benchmark function
    for (int f = 0; f < numFunctions; ++f) {
        // Output metrics
        vector<double> execTimes(30);
        vector<double> bestResults(30);
        vector<vector<double>> meanBestValues(30);

        cout << "Running firefly algorithm for benchmark function " << f + 1 << endl;
        for (int run = 0; run < 30; ++run) {
            auto start = high_resolution_clock::now();

            vector<double> results;
            vector<double> meanBestPerGeneration;
            fireflyAlgorithm(n, d, maxGenerations, numThreads, benchmarkFunctions[f], results, meanBestPerGeneration);
            bestResults[run] = results[0];  // Save only the best solution from each run
            meanBestValues[run] = meanBestPerGeneration;

            auto end = high_resolution_clock::now();
            duration<double> duration = end - start;
            execTimes[run] = duration.count();

            cout << "Run " << run + 1 << " completed." << endl;
            cout << "Best value after this run: " << results[0] << endl;
        }

        // Display results
        double averageBestSolution = accumulate(bestResults.begin(), bestResults.end(), 0.0) / bestResults.size();
        double averageExecutionTime = accumulate(execTimes.begin(), execTimes.end(), 0.0) / execTimes.size();

        cout << "Benchmark function " << f + 1 << " results:" << endl;
        cout << "Average best solution (over 30 runs): " << averageBestSolution << endl;
        cout << "Average execution time: " << averageExecutionTime << " seconds" << endl;
        cout << "=====================================" << endl;
    }

    return 0;
}
