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
    ios_base::sync_with_stdio(0);  // safe, we don't use stdio
    MPI::Init(argc, argv);
    Flags::procs = COMM_WORLD.Get_size();
    Flags::rank = COMM_WORLD.Get_rank();
    SparseMatrix::initElemType();
    if (!Flags::parseArgv(argc, argv))
        return mpi_return(3, "exiting");

    // ------- read CSR --------
    SparseMatrix A;
    if (isMainProcess()) if (readSparseMatrix(Flags::sparse_filename, A)) {
        Flags::size = A.height;  // the matrices are square and equal in size
        ONE_DBG cerr << "---- whole A ----" << endl << A;
    }
    COMM_WORLD.Bcast(&Flags::size, 1, MPI::INT, MAIN_PROCESS);  // All processes need the size
    if (Flags::size == Flags::NOT_SET)
        return mpi_return(4, "failed to read sparse matrix, exiting");
    initPartSizes();
    initGroupComms();

    // ------ scatter data -------
    COMM_WORLD.Barrier();
    double comm_start =  MPI::Wtime();
    DenseMatrix B = generateBFragment();
    ONE_DBG cerr << "---- B part ----" << endl << B;
    vector<int> nnzs;
    A = splitAndScatter(A, nnzs);
    ONE_DBG cerr << "---- A part ----" << endl << A;
    if (Flags::repl > 1) {
        replicateA(A, nnzs);
        ONE_DBG cerr << "---- A part after repl ----" << endl << A;
    }
    COMM_WORLD.Barrier();
    double comm_end = MPI::Wtime();
    ONE_WORKER cerr << "Initial communication time: " << fixed << (comm_end -  comm_start) << "s" << endl;

    // ------- compute C -------
    Multiplicator mult(A, B, nnzs);
    double comp_start = MPI::Wtime();
    DenseMatrix C;
    if (Flags::use_inner) {
        C = mult.matmulInnerABC(Flags::exponent, Flags::repl);
    } else {
        C = mult.matmulColumnA(Flags::exponent, Flags::repl);
    }
    COMM_WORLD.Barrier();
    double comp_end = MPI::Wtime();
    ONE_WORKER cerr << "Computation time: " << fixed << (comp_end - comp_start) << "s" << endl;

    // ------- output results -------
    if (Flags::show_results) {
        if (DEBUG) {
            DenseMatrix whole_B = gatherAndShow(B);
            ONE_DBG cerr << "---- B ----" << endl << whole_B;
        }
        DenseMatrix whole_C = gatherAndShow(C);
        if (DEBUG) {  // in debug mode, output everything to stderr to avoid stream mixing
            ONE_DBG cerr << "---- C ----" << endl << whole_C;
        } else {
            ONE_WORKER cout << whole_C.height << " " << whole_C.width << endl;
            ONE_WORKER cout << whole_C;
        }
    }
    if (Flags::count_ge) {
        int ge_elems = reduceGeElems(C, Flags::ge_element);
        if (DEBUG) {  // in debug mode, output everything to stderr to avoid stream mixing
            ONE_DBG cerr << "---- elems >= " << fixed << setprecision(5) << Flags::ge_element
                << " ----" << endl << ge_elems << endl;
        } else {
            ONE_WORKER cout << ge_elems << endl;
        }
    }

    return mpi_return(0);
}
