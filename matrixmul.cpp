#include <mpi.h>
#include <cassert>
#include <iostream>

#include "densematgen.h"
#include "matrix.hpp"
#include "matrix_utils.hpp"
#include "utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


// Extract parts of the code for readability
DenseMatrix generateBFragment();
void splitAndScatterA();


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
    Flags::size = A.height;  // the matrices are square and equal in size
    COMM_WORLD.Bcast(&Flags::size, 1, MPI_INT, MAIN_PROCESS);

    // ------ scatter data -------
    double comm_start =  MPI::Wtime();
    DenseMatrix B = generateBFragment();
    splitAndScatterA();
    COMM_WORLD.Barrier();
    double comm_end = MPI::Wtime();
    if (isMainProcess())
        cerr << "Initial communication time: " << fixed << (comm_end -  comm_start) << "s" << endl;

    // ------- compute C -------
    double comp_start = MPI::Wtime();
    // FIXME: compute C = A ( A ... (AB ) )
    COMM_WORLD.Barrier();
    double comp_end = MPI::Wtime();
    if (isMainProcess())
        cerr << "Computation time: " << fixed << (comp_end - comp_start) << "s" << endl;

    if (Flags::show_results) {
        // FIXME: replace the following line: print the whole result matrix
        DBG {
            if (isMainProcess()) cerr << "---- B ----" << endl;
            gatherAndShow(B);
        }
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


void splitAndScatterA() {
}
