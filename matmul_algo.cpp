#include "matmul_algo.hpp"
#include <algorithm>
#include <mpi.h>

#include "matrix_utils.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


// FIXME this whole file should be a class to have state
static int my_part = -1;


DenseMatrix matmulInnerABC(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication) {
    throw ShouldNotBeCalled("innerABC not yet implemented");
}


static void mulColA(const SparseMatrix &A, const DenseMatrix &B, DenseMatrix &C);
static SparseMatrix rotateColA(const SparseMatrix &A, vector<int> &nnzs);

DenseMatrix matmulColumnA(SparseMatrix &A, DenseMatrix &B,
        int exponent, int replication,
        vector<int> &nnzs) {
    DenseMatrix C(B.height, B.width);
    //if (isMainProcess()) { DBG cerr << "initial sparse fragment: " << endl << A; }
    for (int iter = 0; iter < exponent; ++iter) {
        // FIXME add replication
        for (int part = 0; part < Flags::procs; ++part) {
            mulColA(A, B, C);
            //DBG cerr << "after multiplication " << part << ": " << endl << C;
            // no need for rotation after last part, impossible to rotate with one process
            if (Flags::procs > 1 && part != Flags::procs-1) {
                A = rotateColA(A, nnzs);
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


static void mulColA(const SparseMatrix &A, const DenseMatrix &B, DenseMatrix &C) {
    if (my_part == -1) {
        my_part = Flags::rank;
    }
    // TODO replication
    int part_offset = firstIdxForProcess(Flags::size, Flags::procs, my_part);
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


static SparseMatrix rotateColA(const SparseMatrix &A, vector<int> &nnzs) {
    static vector<double> recv_a_v, send_a_v;
    static vector<int> recv_ij_v, send_ij_v;
    if (recv_a_v.empty()) {
        // Resize the communication vectors once so we don't have to worry later
        // -- it would be of this size at some point anyway
        int max_nnz = *max_element(nnzs.begin(), nnzs.end());
        recv_a_v.resize(max_nnz);
        recv_ij_v.resize(max_nnz + Flags::size + 1);
        send_a_v.reserve(max_nnz);
        recv_ij_v.reserve(max_nnz + Flags::size + 1);
    }
    // TODO use replication group communicator
    int next = (Flags::rank == Flags::procs - 1) ? 0 : Flags::rank + 1;
    int prev = (Flags::rank == 0) ? Flags::procs - 1 : Flags::rank - 1;
    send_a_v.clear();
    send_ij_v.clear();
    A.appendToVectors(send_a_v, send_ij_v);
    //DBG cerr << "send_a_v: " << send_a_v.size() << "  send_ij_v: " << send_ij_v.size()
        //<< "  nnz: " << A.nnz << "  ij: " << A.nnz + A.height + 1
        //<< "  prev nnz: " << nnzs[prev] << "  prev ij: " << nnzs[prev] + A.height + 1
        //<< endl;
    COMM_WORLD.Isend(send_a_v.data(), A.nnz, MPI::DOUBLE, next, ROTATE_SPARSE_A);
    COMM_WORLD.Isend(send_ij_v.data(), A.nnz + A.height + 1, MPI::INT, next, ROTATE_SPARSE_IJ);
    // TODO measure if/how much IRecv is better here
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
    my_part = (my_part == 0) ? Flags::procs - 1 : my_part - 1;
    // FIXME width can differ, but for now it's not used in sparse matrix
    return SparseMatrix(A.height, A.width, nnzs[Flags::rank], recv_a_v.begin(), recv_ij_v.begin());
}
