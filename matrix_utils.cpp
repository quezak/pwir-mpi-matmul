#include "matrix_utils.hpp"
#include <fstream>
#include <iostream>

#include "matrix.hpp"

using namespace std;


int firstIdxForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) * rank + (rank > numSmaller ? rank - numSmaller : 0);
}


int elemsForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank > numSmaller ? 1 : 0);
}


bool readSparseMatrix(const string &filename, SparseMatrix &matrix) {
    ifstream input(filename);
    if (!input.is_open()) {
        cerr << "could not open file: " << filename << endl;
        return false;
    }
    input >> matrix;
    return true;
}
