#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "matrix_utils.hpp"


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
const double& SparseMatrix::at(int row, int col) const
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

    if(it == last_in_ja){
        return SparseMatrix::zero;
    }

    return this->a[first_elem + (it - first_in_ja)];

}

const double SparseMatrix::zero = 0.0;