#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <iostream>
#include <functional>

#include "matrix_utils.hpp"

using std::vector;
using std::istream;
using std::function;


/// Abstract interface for both dense and sparse matrices.
class Matrix {
public:
    int height, width;

    Matrix(int h, int w): height(h), width(w) {}
    Matrix(): Matrix(0, 0) {}
    Matrix(const Matrix &m): height(m.height), width(m.width) {}

    /// Return value at given coordinates.
    virtual double& at(int row, int col) = 0;
    virtual const double& at(int row, int col) const = 0;
    virtual double get(int row, int col) const = 0;

    class Row {
        Matrix *m;
        const int index;
    public:
        Row(Matrix *_m, int row): m(_m), index(row) {}
        double& operator[] (int col) { return m->at(index, col); }
        const double& operator[] (int col) const { return m->at(index, col); }
    };

    /// Syntactic sugar: m[row][col] returns at(row, col)
    virtual Row operator[] (int row) {
        return Row(this, row);
    }

    virtual Matrix& operator= (const Matrix &m) {
        height = m.height;
        width = m.width;
        return *this;
    }

};


class DenseMatrix : public Matrix {
private:
    vector<vector<double>> data;

public:
    typedef function<double(int, int, int)> MatrixGenerator;

    DenseMatrix(int h, int w);
    DenseMatrix(int h, int w, MatrixGenerator gen, int seed);
    DenseMatrix(): DenseMatrix(0, 0) {}
    DenseMatrix(const DenseMatrix &m): Matrix(m), data(m.data) {}

    double& at(int row, int col) override {
        return data[row][col];
    }

    const double& at(int row, int col) const override {
        return data[row][col];
    }

    double get(int row, int col) const override {
        return data[row][col];
    }

    DenseMatrix& operator= (const DenseMatrix &m) {
        Matrix::operator=(m);
        data = m.data;
        return *this;
    }
};


class SparseMatrix : public Matrix {
public:
    int nnz;  // number of nonzero elements
    int max_row_nnz;

    SparseMatrix() : Matrix() {};
    SparseMatrix(int h, int w): Matrix(h, w) {}

    /// Vector definitions available at https://en.wikipedia.org/wiki/Sparse_matrix
    vector<double> a;
    vector<int> ia, ja;

    virtual double& at(int row, int col) override {
        throw ShouldNotBeCalled("at in SparseMatrix");
    }

    virtual const double& at(int row, int col) const override {
        throw ShouldNotBeCalled("at in SparseMatrix");
    }

    virtual double get(int row, int col) const override;

    friend istream& operator>> (istream& input, SparseMatrix& matrix);

    /// Return matrix slice containing rows [start, end)
    SparseMatrix getRowBlock(int start, int end) const;

    /// Return matrix slice containing columns [start, end)
    SparseMatrix getColBlock(int start, int end) const;
};

#endif  // MATRIX_HPP
