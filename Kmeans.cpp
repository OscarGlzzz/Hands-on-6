#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>  // Agregar esta línea para incluir la cabecera <algorithm>
#include <numeric>
#include <random>

using namespace std;

class KMeans {
private:
    int k;  // Número de clusters
    int max_iterations;  // Número máximo de iteraciones
    vector<vector<double>> data;  // Datos de entrada
    vector<int> labels;  // Etiquetas de cluster para cada punto
    vector<vector<double>> centroids;  // Centroides de los clusters

public:
    KMeans(int k, int max_iterations, const vector<vector<double>>& data)
        : k(k), max_iterations(max_iterations), data(data) {}

    // Inicializar centroides de manera aleatoria
    void initializeCentroids() {
        centroids.clear();
        vector<int> indices(data.size());
        iota(indices.begin(), indices.end(), 0);  // Llena el vector con 0, 1, 2, ..., data.size()-1
        shuffle(indices.begin(), indices.end(), default_random_engine{});  // Mezcla aleatoriamente los índices

        for (int i = 0; i < k; ++i) {
            centroids.push_back(data[indices[i]]);
        }
    }

    // Asignar cada punto al cluster más cercano
    void assignPointsToClusters() {
        labels.clear();
        for (const auto& point : data) {
            double min_distance = numeric_limits<double>::max();
            int min_cluster = -1;

            for (int i = 0; i < k; ++i) {
                double distance = calculateDistance(point, centroids[i]);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_cluster = i;
                }
            }

            labels.push_back(min_cluster);
        }
    }

    // Actualizar los centroides basados en los puntos asignados a cada cluster
    void updateCentroids() {
        vector<vector<double>> new_centroids(k, vector<double>(data[0].size(), 0.0));
        vector<int> cluster_sizes(k, 0);

        for (int i = 0; i < data.size(); ++i) {
            int cluster = labels[i];
            cluster_sizes[cluster]++;
            for (int j = 0; j < data[i].size(); ++j) {
                new_centroids[cluster][j] += data[i][j];
            }
        }

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < data[0].size(); ++j) {
                new_centroids[i][j] /= cluster_sizes[i];
            }
        }

        centroids = new_centroids;
    }

    // Ejecutar el algoritmo de k-means
    void runKMeans() {
        initializeCentroids();

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            assignPointsToClusters();
            updateCentroids();
        }
    }

    // Calcular la distancia euclidiana entre dos puntos
    double calculateDistance(const vector<double>& point1, const vector<double>& point2) {
        double distance = 0.0;
        for (int i = 0; i < point1.size(); ++i) {
            distance += pow(point1[i] - point2[i], 2);
        }
        return sqrt(distance);
    }

    // Obtener los centroides finales
    const vector<vector<double>>& getCentroids() const {
        return centroids;
    }

    // Obtener las etiquetas de cluster asignadas a cada punto
    const vector<int>& getLabels() const {
        return labels;
    }
};

int main() {
    // Ejemplo de uso
    vector<vector<double>> input_data = {{1, 2}, {2 , 3}, {3, 1}, {4, 2}, {8, 8}, {9, 1}};
    int k = 3;  // Número de clusters
    int max_iterations = 100;

    KMeans kmeans(k, max_iterations, input_data);
    kmeans.runKMeans();

    const auto& centroids = kmeans.getCentroids();
    const auto& labels = kmeans.getLabels();

    // Imprimir resultados
    for (int i = 0; i < centroids.size(); ++i) {
        cout << "Cluster " << i << " Centroid: ";
        for (int j = 0; j < centroids[i].size(); ++j) {
            cout << centroids[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Labels: ";
    for (int i = 0; i < labels.size(); ++i) {
        cout << labels[i] << " ";
    }
    cout << endl;

    return 0;
}