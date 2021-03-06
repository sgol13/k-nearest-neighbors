#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <vector>

using namespace std;

constexpr int N_FEATURES = 9;
constexpr char TRAINING_DATASET_FILENAME[] = "datasets/training_dataset.txt";
constexpr char VALIDATION_DATASET_FILENAME[] = "datasets/validation_dataset.txt";
constexpr char NEGATIVE_RESULT[] = "type1";
constexpr char POSITIVE_RESULT[] = "type2";
constexpr int N_DIGITS_DATA = 2;
constexpr int N_RANDOM_RECORDS = 10;

class Record {
  private:
    array<int, N_FEATURES> features;
    bool result;

  public:
    // default constructor
    Record();

    // random record constructor, generates random record in given feature ranges
    Record(const array<int, N_FEATURES> &min_range,
           const array<int, N_FEATURES> &max_range);

    // overloaded [] operator - returns value of a feature
    int operator[](int index) const;

    int &operator[](int index);

    // overloaded == operator - checks if all the records' values are equal
    bool operator==(const Record &other) const;

    // returns a real result of a record
    bool realResult() const;

    // predicts a result using training dataset and a given metric function
    bool
    predictResult(const vector<Record> &training_dataset, const int k_parameter,
                  const function<double(const Record &, const Record &)> &metric);

    // overloaded input stream operator
    friend istream &operator>>(istream &stream, Record &record);
};

// METRICS FUNCTIONS
double euclideanDistance(const Record &x, const Record &y);

double manhattanDistance(const Record &x, const Record &y);

double chebyshevDistance(const Record &x, const Record &y);

double railwayDistance(const Record &x, const Record &y);

double hammingDistance(const Record &x, const Record &y);

double correlationDistance(const Record &x, const Record &y);

// OTHER GLOBAL FUNCTIONS

// overloaded input stream operator
istream &operator>>(istream &stream, Record &record);

// overloaded output stream operator
ostream &operator<<(ostream &stream, const Record &record);

// reads records from a file to a given vector, returns true if correctly read
bool readData(const char filename[], vector<Record> &input_records);

int main() {

    srand(time(nullptr));

    // validation tests for a given dataset
    const vector<
        pair<const function<double(const Record &, const Record &)>, const string>>
        metrics = {make_pair(euclideanDistance, "Euclidean"),
                   make_pair(manhattanDistance, "Manhattan"),
                   make_pair(chebyshevDistance, "Chebyshev"),
                   make_pair(railwayDistance, "Railway"),
                   make_pair(hammingDistance, "Hamming"),
                   make_pair(correlationDistance, "Correlation")};

    const vector<int> k_values = {1, 3, 5, 7, 9, 11, 13, 15, 17};

    vector<Record> training_dataset;
    vector<Record> validation_dataset;
    bool correct_file_1 = readData(TRAINING_DATASET_FILENAME, training_dataset);
    bool correct_file_2 = readData(VALIDATION_DATASET_FILENAME, validation_dataset);

    // check if files were correctly read
    if (correct_file_1 == false) {
        cout << "Cannot read file " << TRAINING_DATASET_FILENAME << endl;
        return 1;
    }

    if (correct_file_2 == false) {
        cout << "Cannot read file " << VALIDATION_DATASET_FILENAME << endl;
        return 2;
    }

    // tests for a given dataset
    cout << "\n              ";
    for (auto &k : k_values) {
        cout << "  K = " << left << setw(2) << k;
    }
    cout << endl;

    cout << fixed << setprecision(2);
    for (auto &metric : metrics) {

        cout << " " << setw(13) << left << metric.second;

        for (auto k : k_values) {

            int incorrectly_classified = 0;
            for (auto &record : validation_dataset) {

                bool predicted_result =
                    record.predictResult(training_dataset, k, metric.first);
                if (predicted_result != record.realResult()) {
                    incorrectly_classified++;
                }
            }

            double e = 100 * (double)incorrectly_classified /
                       (double)validation_dataset.size();

            cout << setw(7) << right << e << "%";
        }
        cout << endl;
    }
    cout << "\n";

    // calculating feature ranges
    array<int, N_FEATURES> min_range{};
    for (auto &el : min_range)
        el = 10000;

    array<int, N_FEATURES> max_range{};

    for (auto &record : training_dataset) {
        for (int i = 0; i < N_FEATURES; i++) {

            min_range[i] = min(min_range[i], record[i]);
            max_range[i] = max(max_range[i], record[i]);
        }
    }

    return 0;
}

// RECORD CLASS
// default constructor
Record::Record() : result(false) {

    for (int i = 0; i < N_FEATURES; i++)
        features[i] = 0;
}

// overloaded [] operator - returns value of a feature
int Record::operator[](int index) const { return features[index]; }

int &Record::operator[](int index) { return features[index]; }

// overloaded == operator - checks if all the records' values are equal
bool Record::operator==(const Record &other) const {
    return features == other.features;
}

// returns a real result of a record
bool Record::realResult() const { return result; }

// random record constructor, generates random record in given feature ranges
Record::Record(const array<int, N_FEATURES> &min_range,
               const array<int, N_FEATURES> &max_range) {

    result = rand() % 2;
    for (int i = 0; i < N_FEATURES; i++) {

        int mod = (max_range[i] - min_range[i] + 1);
        int new_value;
        new_value = (rand() % mod);
        if (result == 0)
            new_value /= 2;

        new_value += min_range[i];

        features[i] = new_value;
    }
}

// predicts a result using training dataset and a given metric function
bool Record::predictResult(
    const vector<Record> &training_dataset, const int k_parameter,
    const function<double(const Record &, const Record &)> &metric) {

    // calculate distances for all records from a training dataset
    vector<pair<double, bool>> distances;
    for (int i = 0; i < (int)training_dataset.size(); i++) {

        double dist = metric(*this, training_dataset[i]);
        distances.push_back(make_pair(dist, training_dataset[i].realResult()));
    }

    // check results among k closest feature vectors
    sort(distances.begin(), distances.end());

    int positive_counter = 0;
    for (int i = 0; i < min(k_parameter, static_cast<int>(distances.size())); i++) {
        if (distances[i].second) {
            positive_counter++;
        }
    }

    // return a prevailing result
    return (2 * positive_counter) > k_parameter;
}

// METRICS FUNCTIONS
double euclideanDistance(const Record &x, const Record &y) {

    double sum = 0, difference;
    for (int i = 0; i < N_FEATURES; i++) {

        difference = x[i] - y[i];
        sum += difference * difference;
    }

    return sqrt(sum);
}

double manhattanDistance(const Record &x, const Record &y) {

    double sum = 0;
    for (int i = 0; i < N_FEATURES; i++) {
        sum += abs(x[i] - y[i]);
    }

    return sum;
}

double chebyshevDistance(const Record &x, const Record &y) {

    double maximum = 0, abs_difference;
    for (int i = 0; i < N_FEATURES; i++) {

        abs_difference = abs(x[i] - y[i]);
        maximum = max(maximum, abs_difference);
    }

    return maximum;
}

double railwayDistance(const Record &x, const Record &y) {

    if (x == y) {
        return 0;
    }

    int sumX = 0, sumY = 0;
    for (int i = 0; i < N_FEATURES; i++) {

        sumX += x[i] * x[i];
        sumY += y[i] * y[i];
    }

    return sqrt(sumX) + sqrt(sumY);
}

double hammingDistance(const Record &x, const Record &y) {

    int counter = 0;
    for (int i = 0; i < N_FEATURES; i++) {
        if (x[i] != y[1]) {
            counter++;
        }
    }

    return (double)counter;
}

double correlationDistance(const Record &x, const Record &y) {

    double sumX = 0, sumY = 0;
    for (int i = 0; i < N_FEATURES; i++) {

        sumX += x[i];
        sumY += y[i];
    }

    double meanX = sumX / N_FEATURES;
    double meanY = sumY / N_FEATURES;

    Record xc, yc;
    for (int i = 0; i < N_FEATURES; i++) {

        xc[i] = x[i] - meanX;
        yc[i] = y[i] - meanY;
    }

    double dot_product = 0, sumXC_2 = 0, sumYC_2 = 0;
    for (int i = 0; i < N_FEATURES; i++) {

        dot_product += xc[i] * yc[i];
        sumXC_2 += xc[i] * xc[i];
        sumYC_2 += yc[i] * yc[i];
    }

    double standard_deviation_X = sqrt(sumXC_2 / N_FEATURES);
    double standard_deviation_Y = sqrt(sumYC_2 / N_FEATURES);

    return dot_product / (standard_deviation_X * standard_deviation_Y);
}

// OTHER GLOBAL FUNCTIONS
// overloaded input stream operator
istream &operator>>(istream &stream, Record &record) {

    for (int i = 0; i < N_FEATURES; i++) {
        stream >> record.features[i];
    }

    string result_name;
    stream >> result_name;
    if (result_name == POSITIVE_RESULT)
        record.result = true;
    else
        record.result = false;

    return stream;
}

// overloaded output stream operator
ostream &operator<<(ostream &stream, const Record &record) {

    stream << "[";
    for (int i = 0; i < N_FEATURES - 1; i++) {
        stream << setw(N_DIGITS_DATA) << record[i] << "|";
    }

    stream << setw(N_DIGITS_DATA) << record[N_FEATURES - 1] << "]";

    return stream;
}

// reads records from a file to a given vector, returns true if correctly read
bool readData(const char filename[], vector<Record> &input_records) {

    fstream input_file;
    input_file.open(filename, ios::in);
    Record new_record;

    if (input_file.good()) {

        while (input_file >> new_record) {
            input_records.push_back(new_record);
        }

        input_file.close();
        return true;
    }

    return false;
}