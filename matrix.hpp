#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <vector>
#include <iostream>
#include <functional>

#include "utils.hpp"

using std::vector;
using std::istream;
using std::ostream;
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

    virtual void print(ostream& output) const;

};


class DenseMatrix : public Matrix {
protected:
    /// Data is stored in one vector as a sequence of columns, to make sending data easier.
    vector<double> data;

    int index(int row, int col) const { return col * height + row; }
    int iRow(int index) const { return index / height; }
    int iCol(int index) const { return index % height; }

public:
    typedef function<double(int, int, int)> MatrixGenerator;

    /// Zero-size matrix
    DenseMatrix(): DenseMatrix(0, 0) {}
    /// Empty h x w matrix
    DenseMatrix(int h, int w);
    /// h x w matrix filled with a generator function.
    /// Offsets denote the coordinates of the upper-left corner when generating a submatrix.
    DenseMatrix(int h, int w, MatrixGenerator gen, int seed, int rowOffset, int colOffset);
    /// Copy
    DenseMatrix(const DenseMatrix &m): Matrix(m), data(m.data) {}

    double& at(int row, int col) override {
        return data[index(row, col)];
    }

    const double& at(int row, int col) const override {
        return data[index(row, col)];
    }

    double get(int row, int col) const override {
        return data[index(row, col)];
    }

    DenseMatrix& operator= (const DenseMatrix &m) {
        Matrix::operator=(m);
        data = m.data;
        return *this;
    }

    void print(ostream& output) const override;

    double* rawData() { return data.data(); }

    int elems() { return data.size(); }
};


class SparseMatrix : public Matrix {
protected:
    /// Vector definitions available at 
    // https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_.28CSR.2C_CRS_or_Yale_format.29
    vector<double> a;  // size: nnz
    vector<int> ia, ja;  // ia size: h+1, ja size: nnz

public:
    int nnz;  // number of nonzero elements
    int max_row_nnz;

    /// Zero-size matrix
    SparseMatrix() : Matrix() {}
    /// Empty h x w matrix
    SparseMatrix(int h, int w): Matrix(h, w) {}
    /// Initialize matrix from vectors used for scattering
    SparseMatrix(int h, int w, int _nnz,
            vector<double>::const_iterator a_it,
            vector<int>::const_iterator ij_it);
    /// Copy
    SparseMatrix(const SparseMatrix &m):
        Matrix(m), a(m.a), ia(m.ia), ja(m.ja), nnz(m.nnz), max_row_nnz(m.max_row_nnz) {}

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

    /// Append a submatrix to vectors that can be used for scattering
    void appendToVectors(vector<double>& a_v, vector<int>& a_count_v, vector<int>& a_pos_v,
            vector<int>& ij_v, vector<int>& ij_count_v, vector<int>& ij_pos_v);

    SparseMatrix& operator= (const SparseMatrix &m) {
        Matrix::operator=(m);
        nnz = m.nnz;
        max_row_nnz = m.max_row_nnz;
        a = m.a;
        ia = m.ia;
        ja = m.ja;
        return *this;
    }

    void print(ostream& output) const override;
};

istream& operator>> (istream& input, SparseMatrix& m);
template<class T>
auto operator<< (std::ostream& os, const T& t) -> decltype(t.print(os), os) {
    t.print(os); 
    return os; 
} 

#endif  // MATRIX_HPP
