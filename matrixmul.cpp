#include <mpi.h>
#include <cassert>
#include <iostream>

#include "densematgen.h"
#include "matmul_algo.hpp"
#include "matrix.hpp"
#include "matrix_utils.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


// Extract parts of the code for readability
DenseMatrix generateBFragment();
SparseMatrix splitAndScatter(const SparseMatrix &m);


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
    A = splitAndScatter(A);
    if (DEBUG) {
        DenseMatrix densePartA(A);
        if (isMainProcess()) { DBG cerr << "---- unsparsed unscattered A ----" << endl; }
        gatherAndShow(densePartA);
    }
    COMM_WORLD.Barrier();
    double comm_end = MPI::Wtime();
    if (isMainProcess())
        cerr << "Initial communication time: " << fixed << (comm_end -  comm_start) << "s" << endl;

    // ------- compute C -------
    double comp_start = MPI::Wtime();
    DenseMatrix C;
    if (Flags::use_inner) {
        C = matmulInnerABC(A, B, Flags::exponent, Flags::repl);
    } else {
        C = matmulColumnA(A, B, Flags::exponent, Flags::repl);
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


DenseMatrix generateBFragment() {
    if (Flags::use_inner) {
        throw ShouldNotBeCalled("B generation for innerABC");
    } else {
        return DenseMatrix(Flags::size, idxsForProcess(Flags::size, Flags::procs, Flags::rank),
                generate_double, Flags::gen_seed,
                0, firstIdxForProcess(Flags::size, Flags::procs, Flags::rank));
    }
}
