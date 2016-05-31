#include "multiplicator.hpp"
#include <algorithm>
#include <mpi.h>

#include "matrix_utils.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


DenseMatrix Multiplicator::matmulInnerABC(int exponent, int replication) {
    throw ShouldNotBeCalled("innerABC not yet implemented");
}


DenseMatrix Multiplicator::matmulColumnA(int exponent, int replication) {
    c = replication;
    //if (isMainProcess()) { DBG cerr << "initial sparse fragment: " << endl << A; }
    for (int iter = 0; iter < exponent; ++iter) {
        // FIXME add replication
        for (int part = 0; part < p; ++part) {
            mulColA();
            //if (isMainProcess()) { DBG cerr << "after multiplication " << part << ": " << endl << C; }
            // no need for rotation after last part, impossible to rotate with one process
            if (p > 1 && part != p-1) {
                rotateColA();
                //if (isMainProcess()) { DBG cerr << "after rotation " << part << ": " << endl << A; }
            }
        }
        if (iter != exponent-1) {
            B = C;
            C = DenseMatrix(B.height, B.width);
        }
    }
    return C;
}

Multiplicator::Multiplicator(SparseMatrix &_A, DenseMatrix &_B, vector<int> &_nnzs)
        : p(Flags::procs), n(_B.height), rank(Flags::rank), part_id(Flags::rank),
          A(_A), B(_B), C(_B.height, _B.width), nnzs(_nnzs) {
    int max_nnz = *max_element(nnzs.begin(), nnzs.end());
    // Resize the communication vectors once so we don't have to worry later
    // -- they would be of this size at some point anyway
    recv_a_v.resize(max_nnz);
    recv_ij_v.resize(max_nnz + n + 1);
    // Note resize vs. reserve -- we want the send vectors empty
    send_a_v.reserve(max_nnz);
    send_ij_v.reserve(max_nnz + n + 1);
}


void Multiplicator::mulColA() {
    // TODO replication
    int part_offset = firstIdxForProcess(n, p, part_id);
    // For each sparse submatrix element we have, add value to all C fields in its row
    for (int row = 0; row < A.height; ++row) {
        for (int i = A.ia[row]; i < A.ia[row+1]; ++i) {
            // element == A.a[i]
            // A col == B row - part_offfset == A.ja[i]
            for (int bcol = 0; bcol < B.width; ++bcol) {
                C[row][bcol] += A.a[i] * B[A.ja[i] + part_offset][bcol];
            }
        }
    }
}


void Multiplicator::rotateColA() {
    // TODO use replication group communicator
    int next = (rank == p - 1) ? 0 : rank + 1;
    int prev = (rank == 0) ? p - 1 : rank - 1;
    send_a_v.clear();
    send_ij_v.clear();
    A.appendToVectors(send_a_v, send_ij_v);
    //DBG cerr << "send_a_v: " << send_a_v.size() << "  send_ij_v: " << send_ij_v.size()
        //<< "  nnz: " << A.nnz << "  ij: " << A.nnz + A.height + 1
        //<< "  prev nnz: " << nnzs[prev] << "  prev ij: " << nnzs[prev] + A.height + 1
        //<< endl;
    // Send our part of A to the next process
    COMM_WORLD.Isend(send_a_v.data(), A.nnz, MPI::DOUBLE, next, ROTATE_SPARSE_A);
    COMM_WORLD.Isend(send_ij_v.data(), A.nnz + A.height + 1, MPI::INT, next, ROTATE_SPARSE_IJ);
    // TODO measure if/how much IRecv is better here
    // Receive new part of A from prev process
    MPI::Request req_a = COMM_WORLD.Irecv(recv_a_v.data(), nnzs[prev], MPI::DOUBLE,
            prev, ROTATE_SPARSE_A);
    MPI::Request req_ij = COMM_WORLD.Irecv(recv_ij_v.data(), nnzs[prev] + A.height + 1, MPI::INT,
            prev, ROTATE_SPARSE_IJ);
    req_a.Wait();
    req_ij.Wait();
    // Rotate the nnzs vector by one position to reflect the submatrix rotation
    int tmp = nnzs.back();
    for (int i = 1; i < (int) nnzs.size(); ++i) nnzs[i] = nnzs[i-1];
    nnzs[0] = tmp;
    part_id = (part_id == 0) ? p - 1 : part_id - 1;
    // FIXME width can differ, but for now it's not used in sparse matrix
    // Decode the received part
    A = SparseMatrix(A.height, A.width, nnzs[rank], recv_a_v.begin(), recv_ij_v.begin());
}
