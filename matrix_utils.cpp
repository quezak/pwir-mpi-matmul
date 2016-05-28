#include "matrix_utils.hpp"
#include <fstream>
#include <mpi.h>

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
        COMM_WORLD.Gatherv(m.rawData(), m.elems(), MPI::DOUBLE,
                NULL, NULL, NULL, MPI::DOUBLE,  // recv params are irrelevant for other processes
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


SparseMatrix splitAndScatter(const SparseMatrix &m) {
    int partNnz = 0;
    int n = Flags::size;
    int p = Flags::procs;
    vector<double> a_v;
    vector<int> ij_v;

    if (!isMainProcess()) {
        COMM_WORLD.Scatter(NULL, 0, MPI::INT,
                &partNnz, 1, MPI::INT,
                MAIN_PROCESS);
        a_v.resize(partNnz);
        ij_v.resize(partNnz + n + 1);
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE,
                a_v.data(), partNnz, MPI::DOUBLE,
                MAIN_PROCESS);
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE,
                ij_v.data(), partNnz + n + 1, MPI::DOUBLE,
                MAIN_PROCESS);
    } else {
        vector<double> all_a_v;
        vector<int> all_ij_v;
        vector<int> a_counts, a_displs, ij_counts, ij_displs;
        for (int i = 0; i < p; ++i) {
            int start = firstIdxForProcess(n, p, i);
            int end = start + idxsForProcess(n, p, i);
            SparseMatrix partM = m.getColBlock(start, end);
            partM.appendToVectors(all_a_v, a_counts, a_displs,
                    all_ij_v, ij_counts, ij_displs);
        }
        COMM_WORLD.Scatter(a_counts.data(), 1, MPI::INT,
                &partNnz, 1, MPI::INT,
                MAIN_PROCESS);
        //DBG cerr << "all a size: " << all_a_v.size() << endl;
        //DBG cerr << "a_counts:  " << a_counts;
        //DBG cerr << "a_displs:  " << a_displs;
        //DBG cerr << "all ij size: " << all_ij_v.size() << endl;
        //DBG cerr << "ij_counts:  " << ij_counts;
        //DBG cerr << "ij_displs:  " << ij_displs;
        a_v.resize(partNnz);
        ij_v.resize(partNnz + n + 1);
        COMM_WORLD.Scatterv(all_a_v.data(), a_counts.data(), a_displs.data(), MPI::DOUBLE,
                a_v.data(), partNnz, MPI::DOUBLE,
                MAIN_PROCESS);
        COMM_WORLD.Scatterv(all_ij_v.data(), ij_counts.data(), ij_displs.data(), MPI::INT,
                ij_v.data(), partNnz + n + 1, MPI::INT,
                MAIN_PROCESS);
    }

    int width = idxsForProcess(n, p, Flags::rank);
    return SparseMatrix(n, width, partNnz, a_v.begin(), ij_v.begin());
}
