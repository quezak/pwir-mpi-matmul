#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <string>
#include <functional>
#include <vector>

// TO DO template na rozmiar
#define ROOT 0

struct sizepair
{
    int ja;
    int ia;
    int width;
};

/// Abstract interface for both dense and sparse matrices.
class Matrix
{
public:
    int height = 0, width = 0;

    Matrix() {};
    Matrix(int h, int w): height(h), width(w) {};

    /// Return value at given coordinates.
    virtual double& at(int row, int col) = 0;
    virtual const double& get(int row, int col) const = 0;

    class Row {
        Matrix *m;
        const int index;
    public:
        Row(Matrix *_m, int row): m(_m), index(row) {}
        double& operator[](int col)
        {
            return m->at(index, col);
        }
    };

    /// Syntactic sugar: m[row][col] = v should run set(row, col).
    // note: this probably should hava a const version, but who cares
    virtual Row operator[](int row)
    {
        return Row(this, row);
    }

};

class SparseMatrix : public Matrix
{
public:
    static double const zero;

    int max_non_zero_in_row;

    SparseMatrix() : Matrix() {};
    SparseMatrix(int h, int w) : Matrix(h, w) {};
    SparseMatrix(std::vector<double>& recva,
                 std::vector<int>& recvia,
                 std::vector<int>& recvja,
                 sizepair& sizes);

    /// Vector definitions available at 
    /// https://en.wikipedia.org/wiki/Sparse_matrix
    std::vector<double> a;
    std::vector<int> ia, ja;

    virtual double& at(int row, int col) override;
    virtual const double& get(int row, int col) const override;

    friend std::istream& operator>>(std::istream& input, SparseMatrix& matrix);

    /// Return matrix slice containing rows [start, end)
    SparseMatrix getRowBlock(int start, int end) const;

    /// Return matrix slice containing columns [start, end)
    SparseMatrix getColBlock(int start, int end) const;
};

class DenseMatrix : public Matrix
{
public:
    // 2D array represented as 1D array
    std::vector<double> data;
    typedef std::function<double(int, int, int)> MatrixGenerator;

    DenseMatrix(int h, int w) : Matrix(h, w) {};
    DenseMatrix(int h, int w, MatrixGenerator gen, int seed);

    virtual double& at(int row, int col) override;
    virtual const double& get(int row, int col) const override;
};


class SparseMatrixToSend
{
public:
    std::vector<double> allas;
    std::vector<int> alljas, allias;
    std::vector<int> japositions, iapositions;
    std::vector<int> jaelems, iaelems;
    std::vector<sizepair> sizes; 

    void fill(SparseMatrix& whole, int repl_fact,
              int num_processes);

    int processFirstIndex(int size, int parts, int rank);
    int elemsForProcess(int size, int parts, int rank);

    // returns the colBlock for current process
    SparseMatrix scatterv(std::vector<double>& recva,
                          std::vector<int>& recvia,
                          std::vector<int>& recvja,
                          int num_processes,
                          int repl_fact,
                          int rank);

    template<typename N>
    void replicateVector(std::vector<N>& toReplicate,
                         int end, int times);

    void extendPosition(std::vector<int>& positions, int target_size, int times);
};

void printSparseMatrix(const SparseMatrix& to_print);

#endif  // MATRIX_UTILS_HPP
