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
    parts = p/c;
    ONE_DBG cerr << "  gsize: " << Flags::group_comm.Get_size() << "  parts: " << parts << endl;
    ONE_DBG cerr << "initial sparse fragment: " << endl << A;
    for (int iter = 0; iter < exponent; ++iter) {
        for (int part = 0; part < parts; ++part) {
            mulColA();
            ONE_DBG cerr << "after multiplication " << part << ": " << endl << C;
            // no need for rotation after last part, impossible to rotate with one process
            if (parts > 1 && part != parts-1) {
                rotateColA();
                ONE_DBG cerr << "after rotation " << part << ": " << endl << A;
            }
        }
        if (iter != exponent-1) {
            B = C;
            C = DenseMatrix(B.height, B.width, B.row_off, B.col_off);
        }
    }
    return C;
}


Multiplicator::Multiplicator(SparseMatrix &_A, DenseMatrix &_B, vector<int> &_nnzs)
        : p(Flags::procs), n(_B.height), g_rank(Flags::group_comm.Get_rank()), part_id(groupId()),
          A(_A), B(_B), C(_B.height, _B.width, _B.row_off, _B.col_off), nnzs(_nnzs) {
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
    ONE_DBG cerr << "mul part_id: " << part_id << endl;
    for (const auto &elem : A.values) {  // for each element in sparse matrix
        for (int bcol = 0; bcol < B.width; ++bcol) {  // for each B column we have
            // no offsets, elem.row, elem.col are absolute values in the original matrix
            C[elem.row][bcol] += elem.val * B[elem.col][bcol];
        }
    }
}


void Multiplicator::rotateColA() {
    ONE_DBG cerr << "send part_id: " << part_id << "  width: " << A.width
        << "  nnz: "  << A.nnz() << "  cur nnzs: " << nnzs;
    int next = (g_rank == parts - 1) ? 0 : g_rank + 1;
    int prev = (g_rank == 0) ? parts - 1 : g_rank - 1;
    ONE_DBG cerr << "grank: " << g_rank << "  next: " << next << "  prev: " << prev
        << "  gsize: " << Flags::group_comm.Get_size() << "  parts: " << parts << endl;
    // Rotate the nnzs vector by one position to reflect the submatrix rotation
    //int tmp = nnzs.back();
    //for (int i = nnzs.size()-1; i >= 1; --i) nnzs[i] = nnzs[i-1];
    //nnzs.front() = tmp;
    part_id = (part_id == 0) ? parts - 1 : part_id - 1;
    // Prepare send & recv buffers
    send_A = A;  // TODO check if this can use move-constructor
    A = SparseMatrix(send_A.height, partSize(true, part_id), 0, partStart(true, part_id), nnzs[part_id]);
    ONE_DBG cerr << "recv part_id: " << part_id << "  width: " << A.width
        << "  nnz: " << A.nnz() << "  new nnzs: " << nnzs;
    // Send our part of A to the next process asynchronously
    // TODO check if we need to wait for the previous loop's Isend
    Flags::group_comm.Isend(send_A.values.data(), send_A.nnz(), SparseMatrix::ELEM_TYPE,
            next, ROTATE_SPARSE_BLOCK_COL);
    // Receive next part of A from prev process
    Flags::group_comm.Recv(A.values.data(), A.nnz(), SparseMatrix::ELEM_TYPE,
            prev, ROTATE_SPARSE_BLOCK_COL);
}
