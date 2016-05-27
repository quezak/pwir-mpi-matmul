#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <iostream>
#include <functional>

#include "matrix_utils.hpp"


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

    /// Return matrix slice containing rows [start, end)
    virtual Matrix& getRowBlock(int start, int end) const = 0;

    /// Return matrix slice containing columns [start, end)
    virtual Matrix& getColBlock(int start, int end) const = 0;

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
    typedef std::function<double(int, int, int)> MatrixGenerator;

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

    Matrix& getRowBlock(int start, int end) const override {
        throw ShouldNotBeCalled("getRowBlock");
    }

    Matrix& getColBlock(int start, int end) const override {
        throw ShouldNotBeCalled("getColBlock");
    }

    DenseMatrix& operator= (const DenseMatrix &m) {
        Matrix::operator=(m);
        data = m.data;
        return *this;
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

#endif  // MATRIX_HPP
