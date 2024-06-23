#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <numeric>  
#include <cstdlib>  
#include <numeric>
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

// Schwefel Function
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
    for (size_t i = 0; i < x.size() - 1; ++i) {
        double num = pow(sin(sqrt(x[i] * x[i] + x[i + 1] * x[i + 1])), 2) - 0.5;
        double denom = pow(1 + 0.001 * (x[i] * x[i] + x[i + 1] * x[i + 1]), 2);
        sum += 0.5 + num / denom;
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
    double sum1 = 0.0;
    double sum2 = 0.0;
    int d = x.size();
    for (double xi : x) {
        sum1 += xi * xi;
        sum2 += cos(2 * M_PI * xi);
    }
    return -20.0 * exp(-0.2 * sqrt(sum1 / d)) - exp(sum2 / d) + 20 + M_E;
}

// Schwefel 2.22 Function
double schwefel222Function(const vector<double>& x) {
    double sum = 0.0;
    double prod = 1.0;
    for (double xi : x) {
        sum += abs(xi);
        prod *= abs(xi);
    }
    return sum + prod;
}


//-----------Firefly algoritam----------------
// Generisanje slucajnog broja izmedju min i max
double randomDouble(double min, double max) {
    return min + (max - min) * (rand() / (RAND_MAX + 1.0));
}


void fireflyAlgorithm(int n, int d, int maxGenerations, int numThreads, vector<double>& results, vector<double>& meanBestPerGeneration, double (*benchmarkFunction)(const vector<double>&)) {
    omp_set_num_threads(numThreads);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-0.5, 0.5);  

    vector<vector<double>> fireflies(n, vector<double>(d));
    vector<double> lightIntensity(n);

    // Inicijalizacija populacije
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            fireflies[i][j] = dist(gen);
        }
        lightIntensity[i] = benchmarkFunction(fireflies[i]) + 1e-10;  // Dodajemo malu konstantu
    }

    double alpha = 0.2;
    double beta0 = 0.5;
    double gamma = 0.1;

    vector<double> bestSolution = fireflies[0];
    double bestIntensity = lightIntensity[0];

    int stagnationCounter = 0;
    double prevBestIntensity = DBL_MAX;

    for (int t = 0; t < maxGenerations; ++t) {
        alpha *= 0.99;

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (lightIntensity[j] < lightIntensity[i]) {
                    double r = 0.0;
                    for (int k = 0; k < d; ++k) {
                        r += (fireflies[i][k] - fireflies[j][k]) * (fireflies[i][k] - fireflies[j][k]);
                    }
                    r = sqrt(r);
                    for (int k = 0; k < d; ++k) {
                        double beta = beta0 * exp(-gamma * r * r);
                        fireflies[i][k] += beta * (fireflies[j][k] - fireflies[i][k]) + alpha * (dist(gen) - 0.5);

                        fireflies[i][k] = max(-0.5, min(0.5, fireflies[i][k]));  
                    }

                    vector<double> localBest = fireflies[i];
                    double localBestIntensity = benchmarkFunction(localBest) + 1e-10;
                    for (int k = 0; k < d; ++k) {
                        double temp = localBest[k];
                        localBest[k] += dist(gen) * 0.05;  // Smanjujemo korak lokalne pretrage
                        localBest[k] = max(-0.5, min(0.5, localBest[k])); 
                        double newIntensity = benchmarkFunction(localBest) + 1e-10;
                        if (newIntensity < localBestIntensity) {
                            localBestIntensity = newIntensity;
                        }
                        else {
                            localBest[k] = temp;
                        }
                    }

                    fireflies[i] = localBest;
                    lightIntensity[i] = localBestIntensity;
                }
            }
        }

        int currentBestIndex = min_element(lightIntensity.begin(), lightIntensity.end()) - lightIntensity.begin();
        if (lightIntensity[currentBestIndex] < bestIntensity) {
            bestSolution = fireflies[currentBestIndex];
            bestIntensity = lightIntensity[currentBestIndex];
        }
        else {
            int worstIndex = max_element(lightIntensity.begin(), lightIntensity.end()) - lightIntensity.begin();
            fireflies[worstIndex] = bestSolution;
            lightIntensity[worstIndex] = bestIntensity;
        }

        if (bestIntensity < prevBestIntensity) {
            stagnationCounter = 0;
            prevBestIntensity = bestIntensity;
        }
        else {
            stagnationCounter++;
        }

        if (stagnationCounter > 100) {
            for (int i = n / 2; i < n; ++i) {
                for (int j = 0; j < d; ++j) {
                    fireflies[i][j] = dist(gen);
                }
                lightIntensity[i] = benchmarkFunction(fireflies[i]) + 1e-10;
            }
            stagnationCounter = 0;
        }

        meanBestPerGeneration.push_back(bestIntensity);

        if (t % 100 == 0) {
            cout << "Generacija: " << t << ", Najbolja vrednost: " << bestIntensity << endl;
        }
    }

    results.push_back(bestIntensity);
}

int main() {
    srand(time(0));

    int n = 50;
    int d = 30;
    int maxGenerations = 2000;
    int numThreads = 1;

    vector<double> execTimes(30);
    vector<double> bestResults(30);
    vector<vector<double>> meanBestValues(30);

    cout << "Pokretanje algoritma 30 puta." << endl;
    for (int run = 0; run < 30; ++run) {
        auto start = high_resolution_clock::now();

        vector<double> results;
        vector<double> meanBestPerGeneration;     
        fireflyAlgorithm(n, d, maxGenerations, numThreads, results, meanBestPerGeneration, schwefel221Function);
        bestResults[run] = results[0];
        meanBestValues[run] = meanBestPerGeneration;

        auto end = high_resolution_clock::now();
        duration<double> duration = end - start;
        execTimes[run] = duration.count();

        cout << "Pokretanje broj " << run + 1 << " zavrseno." << endl;
        cout << "Najbolja vrednost nakon ovog pokretanja: " << results[0] << endl;
    }

    cout << "Prosecno najbolje resenje (preko 30 pokretanja): "
        << accumulate(bestResults.begin(), bestResults.end(), 0.0) / bestResults.size() << endl;
    cout << "Prosecno vreme izvrsavanja: "
        << accumulate(execTimes.begin(), execTimes.end(), 0.0) / execTimes.size() << " sekundi" << endl;

    return 0;
}
