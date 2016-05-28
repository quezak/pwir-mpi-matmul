#include "matrix_utils.hpp"
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <vector>

#include "matrix.hpp"
#include "utils.hpp"

using namespace std;
using MPI::COMM_WORLD;


int firstIdxForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) * rank + (rank > numSmaller ? rank - numSmaller : 0);
}


int idxsForProcess(int size, int parts, int rank) {
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank >= numSmaller ? 1 : 0);
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


void gatherAndShow(DenseMatrix &m) {
    // we need to use Gatherv, because the counts can differ if size is not divisible by p
    if (!isMainProcess()) {
        COMM_WORLD.Gatherv(m.rawData(), m.elems(), MPI_DOUBLE,
                NULL, NULL, NULL, MPI_DOUBLE,  // recv params are irrelevant for other processes
                MAIN_PROCESS);
    } else {
        // assuming the matrix is divided into block columns of width idsxForProcess(...)
        const int p = Flags::procs;
        const int n = Flags::size;
        DenseMatrix recvM(n, n);
        vector<int> counts(p);
        for (int i = 0; i < p; ++i)
            counts[i] = n * idxsForProcess(n, p, i);
        vector<int> displs(Flags::procs);
        displs[0] = 0;
        for (int i = 1; i < p; ++i)
            displs[i] = displs[i-1] + counts[i-1];
        COMM_WORLD.Gatherv(m.rawData(), m.elems(), MPI::DOUBLE,
                recvM.rawData(), counts.data(), displs.data(), MPI::DOUBLE,
                MAIN_PROCESS);
        cout << recvM;
    }
}

