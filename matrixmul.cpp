#include <mpi.h>
#include <cassert>
#include <iostream>

#include "matrix.hpp"
#include "matrix_utils.hpp"
#include "multiplicator.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


int main(int argc, char * argv[]) {

    MPI::Init(argc, argv);
    Flags::procs = COMM_WORLD.Get_size();
    Flags::rank = COMM_WORLD.Get_rank();
    if (!Flags::parseArgv(argc, argv)) {
        cerr << "exiting" << endl;
        MPI::Finalize();
        return 3;
    }

    // ------- read CSR --------
    SparseMatrix A;
    if (isMainProcess() && !readSparseMatrix(Flags::sparse_filename, A)) {
        cerr << "failed to read sparse matrix, exiting" << endl;
        MPI::Finalize();
        return 4;
    }
    if (DEBUG && isMainProcess()) {
        DenseMatrix denseA(A);
        DBG cerr << "---- unsparsed A ----" << endl;
        cerr << denseA;
    }
    Flags::size = A.height;  // the matrices are square and equal in size
    COMM_WORLD.Bcast(&Flags::size, 1, MPI::INT, MAIN_PROCESS);  // All processes need the size

    // ------ scatter data -------
    double comm_start =  MPI::Wtime();
    DenseMatrix B = generateBFragment();
    vector<int> nnzs;
    A = splitAndScatter(A, nnzs);
    if (DEBUG) {
        DenseMatrix densePartA(A);
        if (isMainProcess()) { DBG cerr << "---- unsparsed unscattered A ----" << endl; }
        gatherAndShow(densePartA);
    }
    Multiplicator mult(A, B, nnzs);
    COMM_WORLD.Barrier();
    double comm_end = MPI::Wtime();
    if (isMainProcess())
        cerr << "Initial communication time: " << fixed << (comm_end -  comm_start) << "s" << endl;
    

    // ------- compute C -------
    double comp_start = MPI::Wtime();
    DenseMatrix C;
    if (Flags::use_inner) {
        C = mult.matmulInnerABC(Flags::exponent, Flags::repl);
    } else {
        C = mult.matmulColumnA(Flags::exponent, Flags::repl);
    }
    COMM_WORLD.Barrier();
    double comp_end = MPI::Wtime();
    if (isMainProcess())
        cerr << "Computation time: " << fixed << (comp_end - comp_start) << "s" << endl;

    if (Flags::show_results) {
        if (DEBUG) {
            if (isMainProcess()) { DBG cerr << "---- B ----" << endl; }
            gatherAndShow(B);
        }
        if (isMainProcess()) { DBG cerr << "---- C ----" << endl; }
        // TODO if use_inner
        gatherAndShow(C);
    }
    if (Flags::count_ge) {
        // FIXME: replace the following line: count ge elements
    }

    MPI::Finalize();
    return 0;
}
