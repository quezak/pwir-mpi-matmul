#include <algorithm>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <stdexcept>

#include "densematgen.h"
#include "matrix_utils.hpp"


std::istream& operator>>( std::istream& input, SparseMatrix& matrix )
{
    int temp;
    input >> matrix.height >> matrix.width;
    input >> temp >> matrix.max_non_zero_in_row;
    matrix.a.resize( temp );
    matrix.ia.resize( matrix.height + 1 );
    matrix.ja.resize( temp );

    for ( int i=0; i<temp; i++ ) {
        input >> matrix.a[i];
    }

    for ( int i=0; i<( int )matrix.ia.size(  ); i++ ) {
        input >> matrix.ia[i];
    }

    for ( int i=0; i<temp; i++ ) {
        input >> matrix.ja[i];
    }

    return input;
}

void SparseMatrix::set( int row, int col, double value )
{
    throw std::runtime_error( "Thou shall not change values in the CSR matrix." );
}

/// Return value at given coordinates.
const double& SparseMatrix::get( int row, int col ) const
{
    /// Check which elements are stored in row
    int first_elem = this->ia[row];
    int last_elem = this->ia[row + 1];

    /// Binary search for the given element 
    auto first_in_ja = this->ja.begin(  ) + first_elem;
    auto last_in_ja = this->ja.begin(  ) + last_elem;
    auto it = std::lower_bound( first_in_ja,
                               last_in_ja,
                               col );

    if( it == last_in_ja || *it != col ){
        return SparseMatrix::zero;
    }

    return this->a[first_elem + ( it - first_in_ja )];

}

SparseMatrix SparseMatrix::getRowBlock( int start, int end ) const
{
    SparseMatrix result( end - start + 1, this->width );

    // compute new ia matrix
    result.ia.push_back( 0 );
    int first_row_ia_index = this->ia[start];
    for( auto it=this->ia.begin(  ) + start + 1; it!=this->ia.begin(  ) + end + 1; ++it )
        result.ia.push_back( ( *it ) - first_row_ia_index );

    // copy non-zero values
    result.a.resize( result.ia.back(  ) );
    std::copy( this->a.begin(  ) + first_row_ia_index,
              this->a.begin(  ) + first_row_ia_index + result.ia.back(  ),
              result.a.begin(  ) );

    // compute new ja matrix
    result.ja.resize( result.ia.back(  ) );
    std::copy( this->ja.begin(  ) + first_row_ia_index,
              this->ja.begin(  ) + first_row_ia_index + result.ia.back(  ),
              result.ja.begin(  ) );

    return result;
}

SparseMatrix SparseMatrix::getColBlock( int start, int end ) const
{
    SparseMatrix result( this->height, end - start );

    int next_row = 0;
    int already_included = 0;
    for( int i=0; i<( int )this->a.size(  ); ++i )
    {
        while( this->ia[next_row] <= i )
        {
            result.ia.push_back( already_included );
            next_row++;
        }

        if( this->ja[i] >= start && this->ja[i] < end )
        {
            // this element belongs to colBlock
            result.ja.push_back( this->ja[i] - start );
            result.a.push_back( this->a[i] );
            ++already_included;
        }
    }

    // If the last rows are empty, the vector has to be resized
    // the value should consider all non-empty fields in the last row
    result.ia.resize( this->height + 1, result.a.size(  ) );

    return result;
}

const double SparseMatrix::zero = 0.0;

void DenseMatrix::set( int row, int col, double value )
{
    // ordered fortran-like
    this->data.at(height * col + row) = value;
}

void DenseMatrix::increase( int row, int col, double value )
{
    // ordered fortran-like
    this->data.at(height * col + row) += value;
}

void DenseMatrix::resize( int height, int width )
{
    this->data.resize( height * width );
    this->width = width;
    this->height = height;
}

int DenseMatrix::bigger_than( double threshold )
{
    return count_if( data.begin(), data.end(),
                     [ threshold ]( double val ){ return val > threshold; });
}


const double& DenseMatrix::get( int row, int col ) const
{
    // ordered fortran-like
    return this->data[height * col + row];
}

template<typename N>
void SparseMatrixToSend::replicateVector( std::vector<N>& toReplicate,
                                          int end,
                                          int times )
{
    toReplicate.resize( end * times );
    for( int i = 0; i < times; i++ )
    {
        std::copy( toReplicate.begin(  ),
                  toReplicate.begin(  ) + end,
                  toReplicate.begin(  ) + end * i );
    }
}

void SparseMatrixToSend::extendPosition( std::vector<int>& positions,
                                         int target_size,
                                         int times )
{
    int positions_size = positions.size(  );
    positions.resize( positions_size * times );

    for( int i = 0; i < times; i++ )
    {
        for( int j = 0; j < positions_size; j++ )
        {
            positions[i * positions_size + j] = positions[j] + i * target_size;
        }
    }
}

void SparseMatrixToSend::fill( SparseMatrix& whole,
                               int repl_fact,
                               int num_processes,
                               bool inner )
{
    int global_iaposition = 0;
    int global_japosition = 0;

    this->allas.reserve( whole.a.size(  ) );
    this->allias.reserve( whole.ia.size(  ) );
    this->alljas.reserve( whole.ja.size(  ) );

    this->sizes.resize( num_processes );
    this->iapositions.resize( num_processes );
    this->japositions.resize( num_processes );

    int base = 0;
    int true_index = 0;
    for( int i = 0; i < num_processes; ++i )
    {
        int start = Utils::processFirstIndex( whole.width, num_processes, i );
        int end = Utils::processFirstIndex( whole.width, num_processes, i + 1 );

        if( end == 0 )
        {
            end = whole.width;
        }
        SparseMatrix lastPart;

        if( inner ) 
        {
            lastPart = whole.getRowBlock( start, end );
        } else {
            lastPart = whole.getColBlock( start, end );
        } 

        this->iaelems.push_back( lastPart.ia.size(  ) );
        this->jaelems.push_back( lastPart.ja.size(  ) );

        sizepair temp_sizes;
        temp_sizes.ja = lastPart.ja.size(  );
        temp_sizes.ia = lastPart.ia.size(  );
        temp_sizes.width = lastPart.width;
        temp_sizes.whole_width = whole.width;

        this->sizes.at( true_index ) = temp_sizes;
        this->iapositions.at( true_index ) = global_iaposition;
        this->japositions.at( true_index ) = global_japosition;

        global_japosition += lastPart.ja.size(  );
        global_iaposition += lastPart.ia.size(  );

        this->allas.insert( this->allas.end(  ),
                            lastPart.a.begin(  ),
                            lastPart.a.end(  ) );

        this->alljas.insert( this->alljas.end(  ),
                             lastPart.ja.begin(  ),
                             lastPart.ja.end(  ) );

        this->allias.insert( this->allias.end(  ),
                             lastPart.ia.begin(  ),
                             lastPart.ia.end(  ) );

        // Sending order 0, c, 2c, ..., 1, c + 1, 
        true_index += repl_fact;
        if(true_index >= num_processes)
        {
            base += 1;
            true_index = base;
        }

    }
    /* NOPE!
    //copy positions
    this->extendPosition( this->iapositions, this->allias.size(  ), repl_fact );
    this->extendPosition( this->japositions, this->alljas.size(  ), repl_fact );

    this->replicateVector<double>( this->allas, this->allas.size(  ), repl_fact );
    this->replicateVector<int>( this->allias, this->allias.size(  ), repl_fact );
    this->replicateVector<int>( this->alljas, this->alljas.size(  ), repl_fact );
    this->replicateVector<sizepair>( this->sizes, this->sizes.size(  ), repl_fact );
    this->replicateVector<int>( this->jaelems, this->jaelems.size(  ), repl_fact );
    this->replicateVector<int>( this->iaelems, this->iaelems.size(  ), repl_fact );
    */

}


int Utils::processFirstIndex( int size, int parts, int rank )
{
    rank %= parts;
    int numSmaller = parts - ( size % parts );  // number of parts that are smaller by one element
    return ( size / parts ) * rank + ( rank > numSmaller ? rank - numSmaller : 0 );
}

SparseMatrix SparseMatrixToSend::scatterv( std::vector<double>& recva,
                                           std::vector<int>& recvia,
                                           std::vector<int>& recvja,
                                           int num_processes,
                                           int repl_fact,
                                           int rank,
                                           sizepair& part_info,
                                           MPI::Intracomm& comm_scatter )
{

    sizepair my_sizes;

    comm_scatter.Scatter( ( void* )this->sizes.data(  ),
                          4,
                          MPI::INT,
                          ( void* )&my_sizes,
                          4,
                          MPI::INT,
                          ROOT );

    // wniosek - trzeba stworzyć sztuczną grupę, która posegreguje części!!
    // LOL, działa
    /*if(rank == 1)
    {
        std::cout << my_sizes.ja << " " << my_sizes.ia << " " << " " << rank << std::endl;
    } if(rank == 0)
    {
        print_sparse_informatdive(*it);
    }*/

    recva.resize( my_sizes.ja );
    recvia.resize( my_sizes.ia );
    recvja.resize( my_sizes.ja );
    part_info = my_sizes;

    comm_scatter.Scatterv( ( void* )this->allas.data(  ),
                           ( const int* )this->jaelems.data(  ),
                           ( const int* )this->japositions.data(  ),
                           MPI::DOUBLE,
                           ( void* )recva.data(  ),
                           my_sizes.ja,
                           MPI::DOUBLE,
                           ROOT );

    comm_scatter.Scatterv( ( void* )this->alljas.data(  ),
                           ( const int* )this->jaelems.data(  ),
                           ( const int* )this->japositions.data(  ),
                           MPI::INT,
                           ( void* )recvja.data(  ),
                           my_sizes.ja,
                           MPI::INT,
                           ROOT );

    comm_scatter.Scatterv( ( void* )this->allias.data(  ),
                           ( const int* )this->iaelems.data(  ),
                           ( const int* )this->iapositions.data(  ),
                           MPI::INT,
                           ( void* )recvia.data(  ),
                           my_sizes.ia,
                           MPI::INT,
                           ROOT );
    SparseMatrix my_part( recva, recvia, recvja, my_sizes );

    return my_part;

}


int Utils::elemsForProcess( int size, int parts, int rank )
{
    rank %= parts;
    int numSmaller = parts - ( size % parts );  // number of parts that are smaller by one element
    return ( size / parts ) + ( rank > numSmaller ? 1 : 0 );
}

SparseMatrix Utils::merge_sparse_matrices(int axis, std::vector<SparseMatrix> matrices)
{
    /*
    Axis 0 -> |_| | | (join)
    Axis 1 ->  _
               _
               _ (join)
    
    The vector of matrices should be ordered.
    */

    int result_width = 0;
    int result_height = 0;

    std::vector<int> conc_size; // widths or heights

    if( axis == 1 )
    {
        result_width = matrices[0].width;
        result_height = 0;
        for( auto it = matrices.begin(); it != matrices.end(); it++ )
        {
            result_height += it->height;
        }
    } else {
        result_height = matrices[0].height;
        for( auto it = matrices.begin(); it != matrices.end(); it++ )
        {
            result_width += it->width;
            conc_size.push_back( result_width );
        }

    }
    SparseMatrix result( result_height, result_width );

    if( axis == 1)
    {
        result.ia.push_back(0);
        int last_ia = 0;
        for( int mat_idx = 0; mat_idx < ( int ) matrices.size(); mat_idx += 1 )
        {

            std::copy( matrices[mat_idx].a.begin(), matrices[mat_idx].a.end(),
                       back_inserter(result.a));
            std::copy( matrices[mat_idx].ja.begin(), matrices[mat_idx].ja.end(),
                       back_inserter(result.ja));

            for( int idx2 = 1; idx2 < ( int ) matrices[mat_idx].ia.size(); idx2++ )
            {
                result.ia.push_back(matrices[mat_idx].ia[idx2] + last_ia);
            }

            last_ia = result.ia.back();
        }
    } else {
        int ia_size = matrices[0].ia.size();
        result.ia.resize(ia_size, 0);
        std::vector<int> prev_ia, a_offsets;
        prev_ia.resize(ia_size, 0);
        a_offsets.resize(ia_size, 0);
        for( int i = 0; i < ia_size; i++ )
        {
            int current_width = 0;
            for( int mat_idx = 0; mat_idx < ( int ) matrices.size(); mat_idx += 1 )
            {
                int current_ia = matrices[mat_idx].ia[i];
                int to_add = current_ia - prev_ia[mat_idx];
                for( ; to_add > 0; to_add-- )
                {
                   result.a.push_back( matrices[mat_idx].a[a_offsets[mat_idx]]);
                   result.ja.push_back( matrices[mat_idx].ja[a_offsets[mat_idx]] +
                                        current_width );
                   a_offsets[mat_idx]++; 
                }
                result.ia[i] += current_ia;
                prev_ia[mat_idx] = current_ia;
                current_width += matrices[mat_idx].width;
            }
        }

    }

    return result;
}

SparseMatrix::SparseMatrix( std::vector<double>& recva,
                            std::vector<int>& recvia,
                            std::vector<int>& recvja,
                            sizepair& sizes ) : Matrix( sizes.ia - 1, sizes.width ),
                                                a( recva ),
                                                ia( recvia ),
                                                ja( recvja ) {}

DenseMatrix::DenseMatrix( int h, int w ) : Matrix( h, w )
{
    this->data.resize( h*w, 0 );
}


DenseMatrix Generation::get_dense_part( int seed, int col_begin, int col_end, int rows )
{
    // 1. Create matrix

    DenseMatrix result( rows, col_end - col_begin );
    
    // 2. Fill matrix
    for( int i = 0; i < col_end - col_begin; i++ )
    {
        for( int j = 0; j < rows; j++ )
        {
            int col_index = i + col_begin;
            result.set( j, i, generate_double( seed, j, col_index ) );
        }
    }

    return result;
}

void print_sparse_matrix( const SparseMatrix& to_print )
{

    int currentElem = 0;

    std::cout << "HEIGHT " << to_print.height << std::endl;
    std::cout << "WIDTH " << to_print.width << std::endl;
    std::cout << "non zero elements " << to_print.a.size(  ) << std::endl << std::endl;

    for( int i = 0; i < to_print.height; i++ )
    {
        for( int j = 0; j < to_print.width; j++ )
        {
            if( to_print.ia[i + 1] > currentElem && to_print.ja[currentElem] == j )
            {
                std::cout << to_print.a[currentElem];
                currentElem += 1;
            } else {
                std::cout << 0;
            }
            std::cout << "\t";
        }
        std::cout << std::endl;
    }
}

void print_sparse_informative( const SparseMatrix& to_print )
{
    std::cout << "A" << std::endl;

    for(int i = 0 ;  i < ( int ) to_print.a.size(); i++ )
        std::cout << to_print.a[i] << " ";
    std::cout << std::endl << "IA" << std::endl;
    for(int i = 0 ;  i < ( int ) to_print.ia.size(); i++ )
        std::cout << to_print.ia[i] << " ";
    std::cout << std::endl << "JA" << std::endl;
    for(int i = 0 ;  i < ( int ) to_print.ja.size(); i++ )
        std::cout << to_print.ja[i] << " "; 
    std::cout << std::endl;     
}

void print_dense_matrix( const DenseMatrix& to_print, bool condensed )
{
    if( !condensed )
    {
        std::cout << "HEIGHT " << to_print.height << std::endl;
        std::cout << "WIDTH " << to_print.width << std::endl;
    }

    for( int i = 0; i < to_print.height; i++ )
    {
        for( int j = 0; j < to_print.width; j++ )
        {
            std::cout <<  to_print.get( i, j ) << "\t";
        }
        std::cout << std::endl;
    }

}

SparseMatrix InitSenders::allgatherv( SparseMatrix& to_send, int group_size,
                                      MPI::Intracomm& communicator, int rank,
                                      bool use_inner )
{
    // Firstly send sizes (and widths)
    // TO DO - generalize

    int sizes[3]{ ( int ) to_send.ja.size( ), ( int ) to_send.ia.size( ),
                  use_inner ? to_send.height : to_send.width };
    int all_sizes[group_size * 3];

    int ia_displ[group_size], ja_displ[group_size];
    int ia_sizes[group_size], ja_sizes[group_size];
    int dims[group_size];


    communicator.Allgather( (void*) &sizes, 3,
                            MPI::INT, (void*) all_sizes,
                            3, MPI::INT);

    // every process recounts the sizes
    int cur_ia_size = 0;
    int cur_ja_size = 0;
    
    for( int i = 0; i < group_size; i++ )
    {
        ia_displ[i] = cur_ia_size;
        ja_displ[i] = cur_ja_size;

        cur_ja_size += all_sizes[3 * i];
        cur_ia_size += all_sizes[3 * i + 1];

        ia_sizes[i] = all_sizes[3 * i + 1];
        ja_sizes[i] = all_sizes[3 * i];

        dims[i] = all_sizes[3 * i + 2];
    }

    std::vector<double> new_as;
    std::vector<int> new_jas, new_ias;

    new_as.resize(cur_ja_size);
    new_jas.resize(cur_ja_size);
    new_ias.resize(cur_ia_size);

    // now that the sizes are known, send the matrices.
    communicator.Allgatherv( ( const void* ) to_send.a.data( ), ( int ) to_send.a.size( ),
                             MPI::DOUBLE, ( void* ) new_as.data( ),
                             ja_sizes, ja_displ,
                             MPI::DOUBLE);

    communicator.Allgatherv( ( const void* ) to_send.ia.data( ), ( int ) to_send.ia.size( ),
                             MPI::INT, ( void* ) new_ias.data( ),
                             ia_sizes, ia_displ,
                             MPI::INT);

    communicator.Allgatherv( ( const void* ) to_send.ja.data( ), ( int ) to_send.ja.size( ),
                             MPI::INT, ( void* ) new_jas.data( ),
                             ja_sizes, ja_displ,
                             MPI::INT);


    // check the result
    /*
    if( rank == 0 )
    {
        std::cout << "A" << std::endl;

        for(int i = 0 ;  i < new_as.size(); i++ )
            std::cout << new_as[i] << " ";
        std::cout << std::endl << "IA" << std::endl;
        for(int i = 0 ;  i < new_ias.size(); i++ )
            std::cout << new_ias[i] << " ";
        std::cout << std::endl << "JA" << std::endl;
        for(int i = 0 ;  i < new_jas.size(); i++ )
            std::cout << new_jas[i] << " ";      
    }
    */

    // compute new matrix

    std::vector<SparseMatrix> to_merge;

    for( int i = 0; i < group_size; i++ )
    {
        std::vector<double> temp_a( new_as.begin() + ja_displ[i],
                                    new_as.begin() + ja_displ[i] + ja_sizes[i]);
        std::vector<int> temp_ia( new_ias.begin() + ia_displ[i],
                                  new_ias.begin() + ia_displ[i] + ia_sizes[i]);
        std::vector<int> temp_ja( new_jas.begin() + ja_displ[i],
                                  new_jas.begin() + ja_displ[i] + ja_sizes[i]);

        // dummy ja and whole_width
        sizepair sizes{ 0,
                        use_inner ? dims[i] + 1 : ( int ) to_send.ia.size(),
                        use_inner ? to_send.width : dims[i],
                        0 };

        to_merge.emplace_back( temp_a, temp_ia, temp_ja, sizes );
    }

    if( use_inner )
        return Utils::merge_sparse_matrices(1, to_merge);
    else
        return Utils::merge_sparse_matrices(0, to_merge);
}

void InitSenders::broadcastB( DenseMatrix& dense_part, MPI::Intracomm& comm,
                              int rank )
{
    comm.Bcast( dense_part.data.data( ), dense_part.data.size(),
                MPI::DOUBLE, ROOT );
}

//////////////////////////////////// BLOCKED COLUMN //////////////////////////////////////////////

void BlockedColumn::compute( const SparseMatrix& sm_part, const DenseMatrix& dm_part,
                             DenseMatrix& result, int& current_height)
{

    // iterate over rows in sparse matrix
    int prev_ia = 0;
    int cur_ja = 0;

    for( int i = 0; i < ( int ) sm_part.ia.size(); i++ )
    {
        int values_here = sm_part.ia[i] - prev_ia;
        while( values_here )
        {
            // there are some values here
            for( int j = 0; j < dm_part.width; j++ ){

                /*
                std::cout << "INCREASING " << i-1 << " " << j << " ";
                std::cout << "M " << sm_part.a[cur_ja] << " ";
                std::cout << dm_part.get( current_height + sm_part.ja[cur_ja], j);
                std::cout << " Current ja " << sm_part.ja[cur_ja] <<" h: " << current_height << std::endl;
                */
                result.increase( i - 1, j,
                                 sm_part.a[cur_ja] *
                                 dm_part.get( current_height + sm_part.ja[cur_ja], j));
            }
            values_here--;
            cur_ja ++;
        } 
        prev_ia = sm_part.ia[i];
    }

    // update current height
    current_height = (current_height + sm_part.width) % dm_part.height;
}

SparseMatrix BlockedColumn::send( const SparseMatrix& to_send, MPI::Intracomm& communicator,
                                  int rank )
{
    int sizes[3]{ ( int ) to_send.ia.size(), ( int ) to_send.ja.size(), to_send.width };
    int new_sizes[3];
    int comm_size = communicator.Get_size();

    int from = ( rank + 1 ) % comm_size;
    int to = ( comm_size + rank - 1 ) % comm_size;

    communicator.Isend( (const void*) sizes, 3,
                        MPI::INT, to,
                        BLOCKED_COLUMN_SIZES);

    MPI::Request req = communicator.Irecv( (void*) new_sizes, 3,
                                           MPI::INT, from,
                                           BLOCKED_COLUMN_SIZES );

    // send matrices
    MPI::Request rsend_a = communicator.Isend( (const void*) to_send.a.data( ),
                                                to_send.a.size(),
                                                MPI::DOUBLE, to, BLOCKED_COLUMN_A );

    MPI::Request rsend_ia = communicator.Isend( (const void*) to_send.ia.data( ),
                                                to_send.ia.size(),
                                                MPI::INT, to, BLOCKED_COLUMN_IA );

    MPI::Request rsend_ja = communicator.Isend( (const void*) to_send.ja.data( ),
                                                to_send.ja.size(),
                                                MPI::INT, to, BLOCKED_COLUMN_JA );

    req.Wait();

    std::vector<double> new_a;
    std::vector<int> new_ia, new_ja;

    new_a.resize( new_sizes[1] );
    new_ia.resize( new_sizes[0] );
    new_ja.resize( new_sizes[1] );

    MPI::Request reqa = communicator.Irecv( (void*) new_a.data( ), new_sizes[1],
                                             MPI::DOUBLE, from,
                                             BLOCKED_COLUMN_A );

    MPI::Request reqia = communicator.Irecv( (void*) new_ia.data( ), new_sizes[0],
                                              MPI::INT, from,
                                              BLOCKED_COLUMN_IA );

    MPI::Request reqja = communicator.Irecv( (void*) new_ja.data( ), new_sizes[1],
                                              MPI::INT, from,
                                              BLOCKED_COLUMN_JA );

    reqa.Wait();
    reqia.Wait();
    reqja.Wait();

    rsend_a.Wait();
    rsend_ia.Wait();
    rsend_ia.Wait();

    sizepair sizes_{ 0, ( int ) new_ia.size(), new_sizes[2], 0 };

    return SparseMatrix( new_a, new_ia, new_ja, sizes_ );
}

void BlockedColumn::replace( DenseMatrix& dm_part, DenseMatrix& result )
{
    // hihihi
    dm_part = result;
    result = DenseMatrix( result.height, result.width );
}

////////////////////////////////////// OUTPUT ////////////////////////////////////////////////////

void Output::print_result( const DenseMatrix& dm_part, int rank, int whole_width,
                           int num_processes )
{
    DenseMatrix for_root;

    if( rank == ROOT )
    {
        for_root = DenseMatrix( dm_part.height, whole_width );
    }

    int displs[num_processes];
    int recvcounts[num_processes];

    if( rank == ROOT )
    {
        for( int i = 0; i < num_processes; i++ )
        {
            int index = Utils::processFirstIndex( whole_width, num_processes, i);
            displs[i] = dm_part.height * index;
            if(i > 0)
                recvcounts[i-1] = ( displs[i] - displs[i-1] );

        }
        recvcounts[num_processes - 1] = whole_width * dm_part.height - displs[num_processes - 1];
    }

    MPI::COMM_WORLD.Gatherv( ( const void* ) dm_part.data.data( ), dm_part.data.size(),
                             MPI::DOUBLE, ( void* ) for_root.data.data( ),
                             recvcounts, displs,
                             MPI::DOUBLE, ROOT);

    if( rank == ROOT )
    {
        print_dense_matrix(for_root, true);
    }

}

void Output::print_bigger_than( double ge_element, int rank, DenseMatrix& dm_part )
{
    int sum;
    
    int inner_sum = dm_part.bigger_than( ge_element );

    MPI::COMM_WORLD.Reduce( ( const void* ) &inner_sum, ( void* ) &sum,
                            1, MPI::INT, MPI::SUM, ROOT);

    if( rank == ROOT )
    {
        std::cout << sum << std::endl;
    }
}