#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <string>
#include <functional>
#include <vector>

/// Abstract interface for both dense and sparse matrices.
class Matrix {
public:
    int height, width;

    Matrix() {};
    Matrix(int h, int w): height(h), width(w) {}

    /// Return value at given coordinates.
    virtual double& at(int row, int col) = 0;
    virtual const double& get(int row, int col) const = 0;

    /// Set value at given coordinates.
    virtual void set(int row, int col, double value) = 0;

    class Row {
        Matrix *m;
        const int index;
    public:
        Row(Matrix *_m, int row): m(_m), index(row) {}
        double& operator[](int col) {
            return m->at(index, col);
        }
    };

    /// Syntactic sugar: m[row][col] = v should run set(row, col).
    // note: this probably should hava a const version, but who cares
    virtual Row operator[](int row) {
        return Row(this, row);
    }

};

class SparseMatrix : public Matrix {
public:
    static double const zero;
 
    int non_zero_elements;
    int max_non_zero_in_row;

    SparseMatrix() : Matrix() {};
    SparseMatrix(int h, int w): Matrix(h, w) {}

    /// Vector definitions available at 
    /// https://en.wikipedia.org/wiki/Sparse_matrix
    std::vector<double> a;
    std::vector<int> ia, ja;

    virtual double& at(int row, int col) override;
    virtual const double& get(int row, int col) const override;
    virtual void set(int row, int col, double value) override;

    friend std::istream& operator>>(std::istream& input, SparseMatrix& matrix);

    /// Return matrix slice containing rows [start, end)
    SparseMatrix getRowBlock(int start, int end) const;

    /// Return matrix slice containing columns [start, end)
    SparseMatrix getColBlock(int start, int end) const;
};


#endif  // MATRIX_UTILS_HPP
