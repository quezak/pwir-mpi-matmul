#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <getopt.h>

#include "densematgen.h"
#include "matrix_utils.hpp"

int main( int argc, char * argv[] ) {
    int show_results = 0;
    int use_inner = 0;
    int gen_seed = -1;
    int repl_fact = 1;
    int new_rank = 0;

    int option = -1;

    double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
    int num_processes = 1;
    int mpi_rank = 0;
    int exponent = 1;
    double ge_element = 0;
    int count_ge = 0;

    bool sparse = false;

    MPI::Intracomm inner_b_colblock;
    int inner_b_colblock_rank;

    MPI::Intracomm inner_b_result_col, inner_b_row_height;
    int inner_b_result_col_rank, inner_b_row_height_rank;
    MPI::Intracomm inner_b_diagonal, inner_b_diagonal_rank;

    MPI::Intracomm comm_scatter;

    MPI::Intracomm blocked_colblock, blocked_init;
    int blocked_colblock_rank, blocked_init_rank;

    MPI_Init( &argc, &argv );

    mpi_rank = MPI::COMM_WORLD.Get_rank();
    num_processes = MPI::COMM_WORLD.Get_size();

    SparseMatrix sm;

    while ( ( option = getopt( argc, argv, "vis:f:c:e:g:" ) ) != -1 ) {
        switch ( option ) {
            case 'v': 
                show_results = 1; 
                break;
            case 'i':
                use_inner = 1;
                break;
            case 'f': 
                if ( ( mpi_rank ) == 0 ) { 
                    std::string filename( optarg );
                    std::ifstream file( filename, std::ios::in );
                    file >> sm;

                    if( sm.height > 0 )
                        sparse = true;
                }
                break;
            case 'c': 
                repl_fact = atoi( optarg );
                break;
            case 's':
                gen_seed = atoi( optarg );
                break;
            case 'e': 
                exponent = atoi( optarg );
                break;
            case 'g': 
                count_ge = 1; 
                ge_element = atof( optarg );
                break;
            default:
                fprintf( stderr, "error parsing argument %c exiting\n", option );
                MPI_Finalize(  );
                return 3;
        }
    }

    if ( ( gen_seed == -1 ) || ( ( mpi_rank == 0 ) && ( !sparse ) ) ) {
        fprintf( stderr, "error: missing seed or sparse matrix file; exiting\n" );
        MPI_Finalize(  );
        return 3;
    }

    SparseMatrixToSend toSend;
    sizepair sizes;

    if( use_inner == 0 )
    {
        int parts = num_processes / repl_fact;
        new_rank = (mpi_rank % parts) * repl_fact + mpi_rank / parts;

        comm_scatter = MPI::COMM_WORLD.Split( 0, new_rank );
        new_rank = comm_scatter.Get_rank();
    } else {
        int npprzezc = num_processes / repl_fact;
        int npprzezc2 = num_processes / ( repl_fact * repl_fact );
        int process_height = ( ( mpi_rank % repl_fact ) * npprzezc2 +
                                 mpi_rank / repl_fact ) %
                             ( num_processes / repl_fact );
        inner_b_result_col = MPI::COMM_WORLD.Split( mpi_rank % repl_fact, process_height );
        inner_b_row_height = MPI::COMM_WORLD.Split( process_height, 0 );

        inner_b_result_col_rank = inner_b_result_col.Get_rank();
        inner_b_row_height_rank = inner_b_row_height.Get_rank();

        new_rank = inner_b_result_col_rank * repl_fact +
                   inner_b_row_height_rank;

        comm_scatter = MPI::COMM_WORLD.Split( 0, new_rank );
    }

    // Firstly, cut A into num_processes pieces
    // TO DO: inner splits vertically blah blah
    if( mpi_rank == ROOT ){
        toSend.fill( sm, 1, num_processes, use_inner );
    }

    std::vector<double> mya;
    std::vector<int> myja, myia;

    // and send a piece to every process

    SparseMatrix sparse_part = toSend.scatterv( mya, myia, myja,
                                                num_processes, 1, new_rank,
                                                sizes, comm_scatter );
    // now every process has his own part of A (halftested)
    DenseMatrix dense_part;

    if( use_inner == 0 )
    {

        // Create groups for sharing columns

        blocked_init = MPI::COMM_WORLD.Split( mpi_rank % ( num_processes / repl_fact ) , 0 );
        blocked_init_rank = blocked_init.Get_rank( );

        blocked_colblock = MPI::COMM_WORLD.Split( mpi_rank / ( num_processes / repl_fact ), 0 );
        blocked_colblock_rank = blocked_colblock.Get_rank( );

        int col_begin = Utils::processFirstIndex( sizes.whole_width,
                                                  num_processes,
                                                  mpi_rank );
        int col_end = Utils::processFirstIndex( sizes.whole_width,
                                                num_processes,
                                                mpi_rank + 1 );
                                                
        if( col_end == 0 )
            col_end = sizes.whole_width;

        dense_part = Generation::get_dense_part( gen_seed,
                                                 col_begin,
                                                 col_end,
                                                 sizes.whole_width );
    } else {

        inner_b_colblock = MPI::COMM_WORLD.Split( mpi_rank / repl_fact, 0 );
        inner_b_colblock_rank = inner_b_colblock.Get_rank();

        int pprzezc = num_processes / repl_fact;
        int rankprzezc = mpi_rank / repl_fact;

        int col_begin = Utils::processFirstIndex( sizes.whole_width,
                                                  pprzezc,
                                                  rankprzezc );
        int col_end = Utils::processFirstIndex( sizes.whole_width,
                                                pprzezc,
                                                rankprzezc + 1 );
                                                
        if( col_end == 0 )
            col_end = sizes.whole_width;

        if( inner_b_colblock_rank == 0 )
        {


            // this part will be broadcasted to the group
            dense_part = Generation::get_dense_part( gen_seed,
                                                     col_begin,
                                                     col_end,
                                                     sizes.whole_width );
        } else {
            dense_part.resize( sizes.whole_width,  (col_end - col_begin) );
        }
    } 
    // now every process has his own part of B (tested)

    MPI_Barrier( MPI::COMM_WORLD );
    comm_start =  MPI_Wtime(  );

    if( use_inner == 1 )
    {

        // Broadcast dense to own group
        InitSenders::broadcastB( dense_part, inner_b_colblock,
                                 inner_b_colblock_rank );

        sparse_part = InitSenders::allgatherv( sparse_part, repl_fact,
                                               inner_b_row_height, mpi_rank,
                                               use_inner );

    } else if( repl_fact > 1 )
    {
        // get the other parts of the colblock of A
        sparse_part = InitSenders::allgatherv( sparse_part, repl_fact,
                                               blocked_init, mpi_rank,
                                               use_inner );
    }

    MPI_Barrier( MPI::COMM_WORLD );
    comm_end = MPI_Wtime(  );

    // TO DO : if na inner ( podziel wysokość przez c )
    DenseMatrix result_part( dense_part.height, dense_part.width );

    // TO DO: użyj rang
    int rank_of_leader = (mpi_rank % ( num_processes / repl_fact ) ) * repl_fact;

    int current_height = Utils::processFirstIndex( sizes.whole_width,
                                                   num_processes,
                                                   rank_of_leader );

    SparseMatrix temp_sparse_part;

    comp_start = MPI_Wtime(  );


    for( int i = 0; i < exponent; i++ )
    {
        for( int i = 0; i < blocked_colblock.Get_size(); i++ )
        {

            BlockedColumn::compute( sparse_part, dense_part, result_part, current_height );

            if( mpi_rank == 0){
               //print_sparse_matrix( sparse_part ) ;
               //print_dense_matrix( dense_part ) ;
               //print_dense_matrix( result_part ) ;
            }

            sparse_part = BlockedColumn::send( sparse_part, blocked_colblock,
                                               blocked_colblock_rank );
            
        }
        BlockedColumn::replace( dense_part, result_part );
    }

    // result is now stored in dense part

    MPI_Barrier( MPI::COMM_WORLD );
    comp_end = MPI_Wtime(  );

    if ( show_results ) { // opcja v
        Output::print_result( dense_part, mpi_rank, sizes.whole_width, num_processes );
    }
    if ( count_ge ) { // opcja g
        Output::print_bigger_than( ge_element, mpi_rank, dense_part );
    }

    MPI_Finalize(   );
    return 0;
}

