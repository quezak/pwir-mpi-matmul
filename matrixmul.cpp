#include <mpi.h>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "matrix.hpp"
#include "matrix_utils.hpp"
#include "multiplicator.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


int mpi_return(int code, const string &msg = "") {
    if (msg != "") ONE_WORKER cerr << msg << endl;
    MPI::Finalize();
    return code;
}


int main(int argc, char * argv[]) {

    // ------- init args and mpi -----
    MPI::Init(argc, argv);
    Flags::procs = COMM_WORLD.Get_size();
    Flags::rank = COMM_WORLD.Get_rank();
    SparseMatrix::initElemType();
    if (!Flags::parseArgv(argc, argv))
        return mpi_return(3, "exiting");

    // ------- read CSR --------
    SparseMatrix A;
    if (isMainProcess() && !readSparseMatrix(Flags::sparse_filename, A))
        return mpi_return(4, "failed to read sparse matrix, exiting");
    if (DEBUG && isMainProcess()) {
        DenseMatrix denseA(A);
        ONE_DBG cerr << "---- unsparsed A ----" << endl << denseA;
    }
    Flags::size = A.height;  // the matrices are square and equal in size
    COMM_WORLD.Bcast(&Flags::size, 1, MPI::INT, MAIN_PROCESS);  // All processes need the size
    initPartSizes();
    initGroupComms();

    return mpi_return(0, "---- STOP ----");

    // ------ scatter data -------
    double comm_start =  MPI::Wtime();
    DenseMatrix B = generateBFragment();
    vector<int> nnzs;
    A = splitAndScatter(A, nnzs);
    if (DEBUG) {
        DenseMatrix densePartA(A);
        ONE_DBG cerr << "---- unsparsed unscattered A ----" << endl;
        gatherAndShow(densePartA);
    }
    if (Flags::repl > 1) {
        ONE_DBG cerr << "---- part before repl ----" << endl << A;
        replicateA(A, nnzs);
        ONE_DBG cerr << "---- part  after repl ----" << endl << A;
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
            ONE_DBG cerr << "---- B ----" << endl;
            gatherAndShow(B);
        }
        ONE_DBG cerr << "---- C ----" << endl;
        // TODO if use_inner
        gatherAndShow(C);
    }
    if (Flags::count_ge) {
        int ge_elems = reduceGeElems(C, Flags::ge_element);
        ONE_DBG cerr << "---- elems >= " << fixed << setprecision(5) << Flags::ge_element
            << " ----" << endl;
        // TODO change stream
        ONE_WORKER cerr << ge_elems << endl;
    }

    return mpi_return(0);
}
