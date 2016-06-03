#include "matrix.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace std;


// ----------------------------------------------------------------------------------------------

DenseMatrix::DenseMatrix(int h, int w, int r_o, int c_o): Matrix(h, w, r_o, c_o) {
    data.resize(h*w, 0.0);
}


DenseMatrix::DenseMatrix(int h, int w, int r_o, int c_o, MatrixGenerator gen, int seed)
        : DenseMatrix(h, w, r_o, c_o) {
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            at(i, j) = gen(seed, i + row_off, j + col_off);
}

DenseMatrix::DenseMatrix(const SparseMatrix &m)
        : DenseMatrix(m.height, m.width, m.row_off, m.col_off) {
    for (const auto &elem : m.values)
        at(elem.row - row_off, elem.col - col_off) = elem.val;
}

// ----------------------------------------------------------------------------------------------

istream& operator>>(istream& input, SparseMatrix& m) {
    input >> m.height >> m.width;
    int nnz;
    int max_row_nnz;  // ignored, not useful for now
    input >> nnz >> max_row_nnz;

    vector<double> a;
    vector<int> ia, ja;
    a.resize(nnz);
    ia.resize(m.height + 1);
    ja.resize(nnz);

    for (int i = 0; i < nnz; i++) input >> a[i];
    for (int i = 0; i < m.height + 1; i++) input >> ia[i];
    for (int i = 0; i < nnz; i++) input >> ja[i];

    m.values.reserve(nnz);
    for (int row = 0; row < m.height; ++row)
        for (int i = ia[row]; i < ia[row+1]; ++i)
            m.values.push_back(SparseMatrix::Elem(a[i], row, ja[i]));
    return input;
}

void Matrix::print(ostream &output) const {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j)
            output << fixed << setprecision(5) << setw(10) << get(i, j);
        output << endl;
    }
}

void SparseMatrix::print(ostream &output) const {
    DBG output << "sparse " << height << "x" << width << ", nnz=" << nnz() << endl;
    Matrix::print(output);
}


void DenseMatrix::print(ostream &output) const {
    DBG output << "dense " << height << " x " << width << endl;
    Matrix::print(output);
}


void SparseMatrix::Elem::print(ostream &output) const {
    output << fixed << setprecision(5) << setw(10) << val
        << " @ (" << row << "," << col << ")";
}

// ----------------------------------------------------------------------------------------------

/// Return value at given coordinates.
double SparseMatrix::get(int row, int col) const {
    // TODO check if we need this to be faster than linear (can be log(size) if we sort the elems)
    for (const Elem &elem : values)
        if (elem.at(row - row_off, col - col_off)) return elem.val;
    return 0.0;
}


SparseMatrix SparseMatrix::getRowBlock(int start, int end) const {
    return getBlockByFunction(end-start, width, start, 0,
            [&](const Elem& elem) { return elem.row >= start && elem.row < end; }
            );
}


SparseMatrix SparseMatrix::getColBlock(int start, int end) const {
    return getBlockByFunction(height, end-start, 0, start,
            [&](const Elem& elem) { return elem.col >= start && elem.col < end; }
            );
}


SparseMatrix SparseMatrix::getBlockByFunction(int h, int w, int r_o, int c_o, ElemPredicate pred) const {
    int new_size = count_if(values.begin(), values.end(), pred);
    SparseMatrix result(h, w, r_o, c_o, new_size);  // now the value vector has the right size from start
    copy_if(values.begin(), values.end(), result.values.begin(), pred);
    return result;
}


MPI::Datatype SparseMatrix::ELEM_TYPE;
void SparseMatrix::initElemType() {
    MPI::Datatype field_types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
    int block_lengths[3] = {1, 1, 1};
    MPI_Aint field_displs[3] = {
        offsetof(Elem, val),
        offsetof(Elem, row),
        offsetof(Elem, col)
    };
    ELEM_TYPE = MPI::Datatype::Create_struct(3,
            block_lengths, field_displs, field_types);
    ELEM_TYPE.Commit();
}

