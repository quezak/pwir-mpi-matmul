#include <mpi.h>
#include <cassert>
#include <iostream>

#include "densematgen.h"
#include "matrix.hpp"
#include "matrix_utils.hpp"
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
    Flags::size = A.height;  // the matrices are square and equal in size

    // ------ scatter data -------
    double comm_start =  MPI::Wtime();
    DenseMatrix B;
    if (Flags::use_inner) {
        throw ShouldNotBeCalled("B generation for innerABC");
    } else {
        B = DenseMatrix(A.height, elemsForProcess(A.height, Flags::procs, Flags::rank),
                generate_double, Flags::gen_seed);
    }
    COMM_WORLD.Barrier();
    double comm_end = MPI::Wtime();
    cerr << "Initial communication time: " << (comm_end -  comm_start) << "s" << endl;

    // ------- compute C -------
    double comp_start = MPI::Wtime();
    // FIXME: compute C = A ( A ... (AB ) )
    COMM_WORLD.Barrier();
    double comp_end = MPI::Wtime();
    cerr << "Computation time: " << (comp_end - comp_start) << "s" << endl;

    if (Flags::show_results) {
        // FIXME: replace the following line: print the whole result matrix
    }
    if (Flags::count_ge) {
        // FIXME: replace the following line: count ge elements
    }

    MPI::Finalize();
    return 0;
}
