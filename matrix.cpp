#include "matrix.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace std;


// ----------------------------------------------------------------------------------------------

DenseMatrix::DenseMatrix(int h, int w): Matrix(h, w) {
    data.resize(h*w);
}


DenseMatrix::DenseMatrix(int h, int w, MatrixGenerator gen, int seed,
        int rowOffset, int colOffset): DenseMatrix(h, w) {
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            at(i, j) = gen(seed, i+rowOffset, j+colOffset);
}

// ----------------------------------------------------------------------------------------------

istream& operator>>(istream& input, SparseMatrix& m) {
    input >> m.height >> m.width;
    input >> m.nnz >> m.max_row_nnz;
    m.a.resize(m.nnz);
    m.ia.resize(m.height + 1);
    m.ja.resize(m.nnz);

    for (int i=0; i<m.nnz; i++) {
        input >> m.a[i];
    }

    for (int i=0; i<(int)m.ia.size(); i++) {
        input >> m.ia[i];
    }

    for (int i=0; i<m.nnz; i++) {
        input >> m.ja[i];
    }

    return input;
}

void Matrix::print(ostream &output) const {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j)
            output << setprecision(4) << setw(6) << get(i, j);
        output << endl;
    }
}

void SparseMatrix::print(ostream &output) const {
    DBG output << "sparse " << height << "x" << width << ", nnz=" << nnz << endl;
    Matrix::print(output);
}


void DenseMatrix::print(ostream &output) const {
    DBG output << "dense " << height << " x " << width << endl;
    Matrix::print(output);
}

// ----------------------------------------------------------------------------------------------

/// Return value at given coordinates.
double SparseMatrix::get(int row, int col) const {
    // Check which elements are stored in row
    int first_elem = ia[row];
    int last_elem = ia[row + 1];

    // Binary search for the given element 
    auto first_in_ja = ja.begin() + first_elem;
    auto last_in_ja = ja.begin() + last_elem;
    auto it = lower_bound(first_in_ja, last_in_ja, col);

    if(it == last_in_ja || *it != col) return 0.;

    return a[first_elem + (it - first_in_ja)];

}


SparseMatrix SparseMatrix::getRowBlock(int start, int end) const {
    SparseMatrix result(end - start + 1, width);

    // compute new ia vector
    result.ia.push_back(0);
    int first_row_ia_index = ia[start];
    for(auto it=ia.begin() + start + 1; it!=ia.begin() + end + 1; ++it)
        result.ia.push_back((*it) - first_row_ia_index);

    // copy non-zero values
    result.a.resize(result.ia.back());
    copy(a.begin() + first_row_ia_index,
            a.begin() + first_row_ia_index + result.ia.back(),
            result.a.begin());

    // compute new ja matrix
    result.ja.resize(result.ia.back());
    copy(ja.begin() + first_row_ia_index,
            ja.begin() + first_row_ia_index + result.ia.back(),
            result.ja.begin());

    return result;
}


SparseMatrix SparseMatrix::getColBlock(int start, int end) const {
    SparseMatrix result(height, end - start + 1);

    int next_row = 0;
    int already_included = 0;
    for(int i=0; i<(int)a.size(); ++i) {
        while(ia[next_row] <= i) {
            result.ia.push_back(already_included);
            next_row++;
        }
        if(ja[i] >= start && ja[i] < end) {  // this element belongs to colBlock
            result.ja.push_back(ja[i] - start);
            result.a.push_back(a[i]);
            ++already_included;
        }
    }
    // If the last rows are empty, the vector has to be resized
    result.ia.resize(height + 1, result.ia.back() + 1);

    return result;
}


void SparseMatrix::appendToVectors(vector<double>& a_v, vector<int>& a_count_v, vector<int>& a_pos_v,
        vector<int>& ij_v, vector<int>& ij_count_v, vector<int>& ij_pos_v) {
    a_pos_v.push_back(a_v.size());  // the a vector for this submatrix starts at this position
    a_count_v.push_back(a.size());
    a_v.insert(a_v.end(), a.begin(), a.end());

    ij_pos_v.push_back(ij_v.size());
    ij_count_v.push_back(ia.size() + ja.size());
    ij_v.insert(ij_v.end(), ia.begin(), ia.end());  // ia first, as it has a fixed size
    ij_v.insert(ij_v.end(), ja.begin(), ja.end());
}


SparseMatrix::SparseMatrix(int h, int w, int _nnz,
        vector<double>::const_iterator a_it,
        vector<int>::const_iterator ij_it): SparseMatrix(h, w) {
    nnz = _nnz;
    a = vector<double>(a_it, a_it + nnz);
    ia = vector<int>(ij_it, ij_it + height + 1);
    max_row_nnz = 0;
    for (int i = 1; i < (int) ia.size(); ++i)
        if (ia[i] - ia[i-1] > max_row_nnz)
            max_row_nnz = ia[i] - ia[i-1];
    ja = vector<int>(ij_it + height + 1, ij_it + height + 1 + nnz);
}
