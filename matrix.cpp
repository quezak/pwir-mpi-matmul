#include "matrix.h"
using namespace std;

DenseMatrix::DenseMatrix(int h, int w): Matrix(h, w), data(h) {
    for (auto row : data) row.resize(w);
}


DenseMatrix::DenseMatrix(int h, int w, MatrixGenerator gen, int seed): DenseMatrix(h, w) {
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            at(i, j) = gen(seed, i, j);
        }
    }
}


std::istream& operator>>(std::istream& input, SparseMatrix& matrix)
{
    input >> matrix.height >> matrix.width;
    input >> matrix.non_zero_elements >> matrix.max_non_zero_in_row;
    matrix.a.resize(matrix.non_zero_elements);
    matrix.ia.resize(matrix.height + 1);
    matrix.ja.resize(matrix.non_zero_elements);

    for (int i=0; i<matrix.non_zero_elements; i++) {
        input >> matrix.a[i];
    }

    for (int i=0; i<(int)matrix.ia.size(); i++) {
        input >> matrix.ia[i];
    }

    for (int i=0; i<matrix.non_zero_elements; i++) {
        input >> matrix.ja[i];
    }

    return input;
}

double& SparseMatrix::at(int row, int col)
{
    throw std::runtime_error("Thou shall not change values in the CSR matrix.");
}

/// Return value at given coordinates.
const double& SparseMatrix::get(int row, int col) const
{
    /// Check which elements are stored in row
    int first_elem = this->ia[row];
    int last_elem = this->ia[row + 1];

    /// Binary search for the given element 
    auto first_in_ja = this->ja.begin() + first_elem;
    auto last_in_ja = this->ja.begin() + last_elem;
    auto it = std::lower_bound(first_in_ja,
                               last_in_ja,
                               col);

    if(it == last_in_ja || *it != col){
        return SparseMatrix::zero;
    }

    return this->a[first_elem + (it - first_in_ja)];

}

void SparseMatrix::set(int row, int col, double value)
{
    throw std::runtime_error("Thou shall not change values in the CSR matrix.");
}

SparseMatrix SparseMatrix::getRowBlock(int start, int end) const
{
    SparseMatrix result(end - start + 1, this->width);

    // compute new ia matrix
    result.ia.push_back(0);
    int first_row_ia_index = this->ia[start];
    for(auto it=this->ia.begin() + start + 1; it!=this->ia.begin() + end + 1; ++it)
        result.ia.push_back((*it) - first_row_ia_index);

    // copy non-zero values
    result.a.resize(result.ia.back());
    std::copy(this->a.begin() + first_row_ia_index,
              this->a.begin() + first_row_ia_index + result.ia.back(),
              result.a.begin());

    // compute new ja matrix
    result.ja.resize(result.ia.back());
    std::copy(this->ja.begin() + first_row_ia_index,
              this->ja.begin() + first_row_ia_index + result.ia.back(),
              result.ja.begin());

    return result;
}

SparseMatrix SparseMatrix::getColBlock(int start, int end) const
{
    SparseMatrix result(this->height, end -start + 1);

    int next_row = 0;
    int already_included = 0;
    for(int i=0; i<(int)this->a.size(); ++i)
    {
        while(this->ia[next_row] <= i)
        {
            result.ia.push_back(already_included);
            next_row++;
        }

        if(this->ja[i] >= start && this->ja[i] < end)
        {
            // this element belongs to colBlock
            result.ja.push_back(this->ja[i] - start);
            result.a.push_back(this->a[i]);
            ++already_included;
        }
    }

    // If the last rows are empty, the vector has to be resized
    result.ia.resize(this->height + 1, result.ia.back() + 1);

    return result;
}

const double SparseMatrix::zero = 0.0;
