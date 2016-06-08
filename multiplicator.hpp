#ifndef MULTIPLICATOR_HPP
#define MULTIPLICATOR_HPP

#include <vector>

#include "matrix.hpp"
using std::vector;


class Multiplicator {
private:
    vector<double> recv_a_v, send_a_v;
    vector<int> recv_ij_v, send_ij_v;
    int c;  // replication
    int p;  // num_processes
    int n;  // main matrix size
    int g_rank = Flags::NOT_SET;  // rank inside the rotation group
    int part_id = Flags::NOT_SET;
    int part_first = Flags::NOT_SET;
    int parts = Flags::NOT_SET;
    SparseMatrix send_A;  // a copy of A for sending, so it can be done in parallel with receiving
    MPI::Request isend_req;
    bool first_isend = true;

    void mulColA();
    void mulInnerABC();
    void rotatePartA();
    void innerGatherC();

public:
    SparseMatrix A;  // A matrix part for each process
    DenseMatrix B;  // B matrix part
    DenseMatrix C;  // result part
    vector<int> nnzs;

    Multiplicator(SparseMatrix &_A, DenseMatrix &_B, vector<int> &nnzs);

    DenseMatrix matmulInnerABC(int exponent, int replication);

    DenseMatrix matmulColumnA(int exponent, int replication);

};


#endif  // MULTIPLICATOR_HPP
