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


class DenseMatrix;
class SparseMatrix;


/// Abstract interface for both dense and sparse matrices.
class Matrix {
public:
    int height, width;
    int row_off, col_off;  // offsets to know a submatrix position

    Matrix(int h, int w, int r_o, int c_o): height(h), width(w), row_off(r_o), col_off(c_o) {}
    Matrix(): Matrix(0, 0, 0, 0) {}
    Matrix(const Matrix &m) = default;

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
    };

    class ConstRow {
        const Matrix *m;
        const int index;
    public:
        ConstRow(const Matrix *_m, int row): m(_m), index(row) {}
        const double& operator[] (int col) const { return m->at(index, col); }
    };

    /// Syntactic sugar: m[row][col] returns at(row, col)
    virtual Row operator[] (int row) {
        return Row(this, row);
    }

    virtual ConstRow operator[] (int row) const {
        return ConstRow(this, row);
    }

    virtual Matrix& operator= (const Matrix &m) {
        height = m.height;
        width = m.width;
        row_off = m.row_off;
        col_off = m.col_off;
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
    DenseMatrix(): DenseMatrix(0, 0, 0, 0) {}
    /// Empty h x w matrix
    DenseMatrix(int h, int w, int r_o, int c_o);
    /// h x w matrix filled with a generator function.
    /// Offsets denote the coordinates of the upper-left corner when generating a submatrix.
    DenseMatrix(int h, int w, int r_o, int c_o, MatrixGenerator gen, int seed);
    /// Copy
    DenseMatrix(const DenseMatrix &m): Matrix(m), data(m.data) {}
    /// Convert sparse to dense for debugging purposes
    explicit DenseMatrix(const SparseMatrix &m);

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

    friend void gatherAndShow(DenseMatrix &m, int parts, MPI::Intracomm &comm);
};


class SparseMatrix : public Matrix {
public:
    struct Elem {
        double val;
        int row, col;

        Elem(double v, int r, int c): val(v), row(r), col(c) {}
        Elem(): Elem(0, 0, 0) {}

        bool at(int r, int c) const {
            return row == r && col == c;
        }

        static bool rowOrder(const Elem &a, const Elem &b) {
            return (a.row == b.row) ? (a.col < b.col) : (a.row < b.row);
        }

        static bool colOrder(const Elem &a, const Elem &b) {
            return (a.col == b.col) ? (a.row < b.row) : (a.col < b.col);
        }

        void print(ostream &output) const;
    };
    vector<Elem> values;

    /// Empty h x w matrix with space reserved for nnz elements
    SparseMatrix(int h, int w, int r_o, int c_o, int nnz): Matrix(h, w, r_o, c_o), values(nnz) {}
    /// Empty h x w matrix without reserved space
    SparseMatrix(int h, int w, int r_o, int c_o): SparseMatrix(h, w, r_o, c_o, 0) {}
    /// Zero-size matrix
    SparseMatrix() : SparseMatrix(0, 0, 0, 0) {}
    /// Copy
    SparseMatrix(const SparseMatrix &m): Matrix(m), values(m.values) {}

    virtual double& at(int row, int col) override {
        throw ShouldNotBeCalled("at in SparseMatrix");
    }

    virtual const double& at(int row, int col) const override {
        throw ShouldNotBeCalled("at in SparseMatrix");
    }

    virtual double get(int row, int col) const override;

    int nnz() const { return values.size(); }

    /// Return matrix slice containing rows [start, end)
    SparseMatrix getRowBlock(int start, int end) const;

    /// Return matrix slice containing columns [start, end)
    SparseMatrix getColBlock(int start, int end) const;

    SparseMatrix& operator= (const SparseMatrix &m) {
        Matrix::operator=(m);
        values = m.values;
        return *this;
    }

    void print(ostream& output) const override;

    static MPI::Datatype ELEM_TYPE;
    // initializes the MPI datatype for sparse matrix elements, should be called by all processes
    static void initElemType();


protected:
    typedef function<bool(Elem)> ElemPredicate;
    SparseMatrix getBlockByFunction(int h, int w, int r_o, int c_o, ElemPredicate pred) const;
};

istream& operator>> (istream& input, SparseMatrix& m);
template<class T>
auto operator<< (std::ostream& os, const T& t) -> decltype(t.print(os), os) {
    t.print(os); 
    return os; 
} 

#endif  // MATRIX_HPP
