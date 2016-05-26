#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <exception>

using std::string;
using std::vector;


/// Abstract interface for both dense and sparse matrices.
class Matrix {
public:
    const int height, width;

    Matrix(int h, int w): height(h), width(w) {}

    /// Return value at given coordinates.
    virtual double& at(int row, int col) = 0;
    virtual const double& at(int row, int col) const = 0;

    /// Set value at given coordinates.
    virtual void set(int row, int col, double value) = 0;

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

};


class Exception: public std::exception {
private:
    string msg;
public:
    Exception(const string &s): msg(s) {}
    const char* what() const noexcept override {
        return msg.c_str();
    }
};


class ShouldNotBeCalled: public Exception {
public:
    ShouldNotBeCalled(const string& s): Exception(s) {}
};


class DenseMatrix : public Matrix {
private:
    vector<vector<double>> data;

public:
    typedef std::function<double(int, int, int)> MatrixGenerator;

    DenseMatrix(int h, int w);
    DenseMatrix(int h, int w, MatrixGenerator gen, int seed);

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
};

#endif  // MATRIX_UTILS_HPP
