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


int main(int argc, char * argv[]) {

    // ------- init args and mpi -----
    MPI::Init(argc, argv);
    Flags::procs = COMM_WORLD.Get_size();
    Flags::rank = COMM_WORLD.Get_rank();
    SparseMatrix::initElemType();
    if (!Flags::parseArgv(argc, argv)) {
        cerr << "exiting" << endl;
        MPI::Finalize();
        return 3;
    }
    if (Flags::repl > 1) {
        // Communicator to replicate data (processes will have the same part of A)
        // Unneded if no replication is done
        int repl_id = Flags::rank % (Flags::procs / Flags::repl);
        Flags::repl_comm = COMM_WORLD.Split(repl_id, Flags::rank);
        ONE_DBG cerr << "repl comm rank: " << Flags::repl_comm.Get_rank() 
            << "  size: " << Flags::repl_comm.Get_size() << endl;
    }
    // Communicator to rotate data (processes will have different parts, and together the whole A)
    // Will be just one comm if c=1
    int group_id = Flags::rank / (Flags::procs / Flags::repl);
    Flags::group_comm = COMM_WORLD.Split(group_id, Flags::rank);
    ONE_DBG cerr << "group comm rank: " << Flags::group_comm.Get_rank() 
        << "  size: " << Flags::group_comm.Get_size() << endl;

    // ------- read CSR --------
    SparseMatrix A;
    if (isMainProcess() && !readSparseMatrix(Flags::sparse_filename, A)) {
        cerr << "failed to read sparse matrix, exiting" << endl;
        MPI::Finalize();
        return 4;
    }
    if (DEBUG && isMainProcess()) {
        DenseMatrix denseA(A);
        ONE_DBG cerr << "---- unsparsed A ----" << endl << denseA;
    }
    Flags::size = A.height;  // the matrices are square and equal in size
    COMM_WORLD.Bcast(&Flags::size, 1, MPI::INT, MAIN_PROCESS);  // All processes need the size
    initPartSizes();

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

    MPI::Finalize();
    return 0;
}
