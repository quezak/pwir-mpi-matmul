#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <algorithm>
#include <string>
#include <functional>
#include <vector>

#define ROOT 0
#define DEBUG 0

enum messages
{
    BLOCKED_COLUMN_SIZES,
    BLOCKED_COLUMN_A,
    BLOCKED_COLUMN_IA,
    BLOCKED_COLUMN_JA
};

struct sizepair
{
    int ja;
    int ia;
    int width;  // this field might change during the execution time
    int whole_width;
};

struct sizepair_condensed
{
    int ja;
    int ia;
};

/// Abstract interface for both dense and sparse matrices.
class Matrix
{
public:
    int height = 0, width = 0;

    Matrix( ) {};
    Matrix( int h, int w ): height( h ), width( w ) {};

    /// Return value at given coordinates.
    virtual void set( int row, int col, double value ) = 0;
    virtual const double& get( int row, int col ) const = 0;
};

class SparseMatrix : public Matrix
{
public:
    static double const zero;

    int max_non_zero_in_row;

    SparseMatrix( ) : Matrix( ) {};
    SparseMatrix( int h, int w ) : Matrix( h, w ) {};
    SparseMatrix( std::vector<double>& recva,
                  std::vector<int>& recvia,
                  std::vector<int>& recvja,
                  sizepair& sizes );

    /// Vector definitions available at 
    /// https://en.wikipedia.org/wiki/Sparse_matrix
    std::vector<double> a;
    std::vector<int> ia, ja;

    virtual void set( int row, int col, double value ) override;
    virtual const double& get( int row, int col ) const override;

    friend std::istream& operator>>( std::istream& input, SparseMatrix& matrix );

    /// Return matrix slice containing rows [start, end )
    SparseMatrix getRowBlock( int start, int end ) const;

    /// Return matrix slice containing columns [start, end )
    SparseMatrix getColBlock( int start, int end ) const;
};

class DenseMatrix : public Matrix
{
public:
    // 2D array represented as 1D array
    std::vector<double> data;

    DenseMatrix(  ) : Matrix(  ) {};
    DenseMatrix( int h, int w );

    virtual void set( int row, int col, double value ) override;
    void increase( int row, int col, double value );
    void resize( int height, int width );
    virtual const double& get( int row, int col ) const override;

    int bigger_than( double threshold );
};


class SparseMatrixToSend
{
public:
    std::vector<double> allas;
    std::vector<int> alljas, allias;
    std::vector<int> japositions, iapositions;
    std::vector<int> jaelems, iaelems;
    std::vector<sizepair> sizes; 

    void fill( SparseMatrix& whole, int repl_fact,
              int num_processes, bool inner );

    // returns the colBlock for current process
    SparseMatrix scatterv( std::vector<double>& recva,
                           std::vector<int>& recvia,
                           std::vector<int>& recvja,
                           int num_processes,
                           int repl_fact,
                           int rank,
                           sizepair& part_info,
                           MPI::Intracomm& comm_scatter );

    template<typename N>
    void replicateVector( std::vector<N>& toReplicate,
                         int end, int times );

    void extendPosition( std::vector<int>& positions, int target_size, int times );
};

namespace Generation
{
    DenseMatrix get_dense_part( int seed, int col_begin, int col_end, int rows );
}

namespace Utils
{
    int processFirstIndex( int size, int parts, int rank );
    int elemsForProcess( int size, int parts, int rank );
    SparseMatrix merge_sparse_matrices(int axis, std::vector<SparseMatrix> matrices);
}

namespace InitSenders
{
    SparseMatrix allgatherv( SparseMatrix& to_send, int group_size,
                             MPI::Intracomm& communicator, int rank,
                             bool use_inner );
    void broadcastB( DenseMatrix& dense_part, MPI::Intracomm& comm,
                     int rank);
}

namespace BlockedColumn
{
    void compute( const SparseMatrix& sm_part, const DenseMatrix& dm_part,
                  DenseMatrix& result, int& current_height );
    SparseMatrix send( const SparseMatrix& to_send, MPI::Intracomm& communicator, int rank );
    void replace( DenseMatrix& dm_part, DenseMatrix& result );
}

namespace Output
{
    void print_result( const DenseMatrix& dm_part, int rank, int whole_width,
                       int num_processes );
    void print_bigger_than( double ge_element, int rank, DenseMatrix& dm_part );

}

void print_dense_matrix( const DenseMatrix& to_print, bool condensed );
void print_sparse_matrix( const SparseMatrix& to_print );
void print_sparse_informative( const SparseMatrix& to_print );

#endif  // MATRIX_UTILS_HPP
