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

// Ciljna funkcija sum squares
double sumOfSquaresFunction(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
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

// Benchmark funkcija Step2
double step2Function(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += pow(floor(xi) + 0.5, 2);
    }
    return sum;
}

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


// Generisanje slucajnog broja izmedju min i max
double randomDouble(double min, double max) {
    return min + (max - min) * (rand() / (RAND_MAX + 1.0));
}

// Firefly algoritam
void fireflyAlgorithm(int n, int d, int maxGenerations, int numThreads, vector<double>& results) {
    // Postavljanje broja niti za OpenMP
    omp_set_num_threads(numThreads);

    // Inicijalizacija generatora slučajnih brojeva
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0, 1); // Normalna (Gausova) raspodela

    vector<vector<double>> fireflies(n, vector<double>(d));
    vector<double> lightIntensity(n);

    // Inicijalizacija populacije
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            fireflies[i][j] = dist(gen); // Koristimo normalnu raspodelu
        }
        // Izračunavanje intenziteta svetlosti za svakog svitka
        lightIntensity[i] = step2Function(fireflies[i]);
    }

    // Parametri algoritma
    double alpha = 0.5; // Povećali smo alpha
    double beta0 = 1.0;
    double gamma = 0.01; // Smanjili smo gamma

    // Evolucija kroz generacije
    for (int t = 0; t < maxGenerations; ++t) {
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
                        fireflies[i][k] += beta * (fireflies[j][k] - fireflies[i][k]) + alpha * dist(gen); // Koristimo normalnu raspodelu
                    }
                    lightIntensity[i] = step2Function(fireflies[i]);
                }
            }
        }
        if (t % 100 == 0) {
            cout << "Generacija: " << t << endl;
        }
    }

    // Pronalaženje najboljeg rešenja
    int bestIndex = min_element(lightIntensity.begin(), lightIntensity.end()) - lightIntensity.begin();
    results.push_back(lightIntensity[bestIndex]);
}

int main() {
    srand(time(0));  // Inicijalizacija random seed-a

    int n = 50;  // Broj svitaca
    int d = 30;  // Dimenzija problema
    int maxGenerations = 1000;  // Maksimalan broj generacija

    int numThreads = 6;  // Broj niti

    // Izlazne metrike
    vector<double> execTimes(30);
    vector<double> bestResults(30);

    cout << "Pokretanje algoritma 30 puta." << endl;
    for (int run = 0; run < 30; ++run) {
        auto start = high_resolution_clock::now();

        vector<double> results;
        fireflyAlgorithm(n, d, maxGenerations, numThreads, results);
        bestResults[run] = results[0];  // Čuvamo samo najbolje rešenje iz svakog pokretanja

        auto end = high_resolution_clock::now();
        duration<double> duration = end - start;
        execTimes[run] = duration.count();

        cout << "Pokretanje broj " << run + 1 << " zavrseno." << endl;
        cout << "Srednja najbolja vrednost nakon ovog pokretanja: " << accumulate(results.begin(), results.end(), 0.0) / results.size() << endl;
    }

    // Prikaz rezultata
    cout << "Prosecno najbolje resenje (preko 30 pokretanja): "
        << accumulate(bestResults.begin(), bestResults.end(), 0.0) / bestResults.size() << endl;
    cout << "Prosecno vreme izvrsavanja: "
        << accumulate(execTimes.begin(), execTimes.end(), 0.0) / execTimes.size() << " sekundi" << endl;

    return 0;
}