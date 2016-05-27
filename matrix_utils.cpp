#include <algorithm>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <stdexcept>

#include "matrix_utils.hpp"


std::istream& operator>>(std::istream& input, SparseMatrix& matrix)
{
    int temp;
    input >> matrix.height >> matrix.width;
    input >> temp >> matrix.max_non_zero_in_row;
    matrix.a.resize(temp);
    matrix.ia.resize(matrix.height + 1);
    matrix.ja.resize(temp);

    for (int i=0; i<temp; i++) {
        input >> matrix.a[i];
    }

    for (int i=0; i<(int)matrix.ia.size(); i++) {
        input >> matrix.ia[i];
    }

    for (int i=0; i<temp; i++) {
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
    SparseMatrix result(this->height, end - start);

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
    // the value should consider all non-empty fields in the last row
    result.ia.resize(this->height + 1, result.a.size());

    return result;
}

const double SparseMatrix::zero = 0.0;

double& DenseMatrix::at(int row, int col)
{
    return data[height * col + row];
}

const double& DenseMatrix::get(int row, int col) const
{
    return data[height * col + row];
}

template<typename N>
void SparseMatrixToSend::replicateVector(std::vector<N>& toReplicate,
                                         int end,
                                         int times)
{
    toReplicate.resize(end * times);
    for(int i = 0; i < times; i++)
    {
        std::copy(toReplicate.begin(),
                  toReplicate.begin() + end,
                  toReplicate.begin() + end * i);
    }
}

void SparseMatrixToSend::extendPosition(std::vector<int>& positions,
                                        int target_size,
                                        int times)
{
    int positions_size = positions.size();
    positions.resize(positions_size * times);

    for(int i = 0; i < times; i++)
    {
        for(int j = 0; j < positions_size; j++)
        {
            positions[i * positions_size + j] = positions[j] + i * target_size;
        }
    }
}

void SparseMatrixToSend::fill(SparseMatrix& whole,
                              int repl_fact,
                              int num_processes)
{
    int parts = num_processes / repl_fact;
    int global_iaposition = 0;
    int global_japosition = 0;

    this->allas.reserve(whole.a.size());
    this->allias.reserve(whole.ia.size());
    this->alljas.reserve(whole.ja.size());

    for(int i = 0; i < parts; ++i)
    {
        int start = this->processFirstIndex(whole.width, parts, i);
        int end = this->processFirstIndex(whole.width, parts, i + 1);

        if(end == 0)
        {
            end = whole.width;
        }
        SparseMatrix lastPart = whole.getColBlock(start, end);

        this->iaelems.push_back(lastPart.ia.size());
        this->jaelems.push_back(lastPart.ja.size());

        sizepair temp_sizes;
        temp_sizes.ja = lastPart.ja.size();
        temp_sizes.ia = lastPart.ia.size();
        temp_sizes.width = lastPart.width;

        this->sizes.push_back(temp_sizes);
        this->iapositions.push_back(global_iaposition);
        this->japositions.push_back(global_japosition);

        global_japosition += lastPart.ja.size();
        global_iaposition += lastPart.ia.size();

        this->allas.insert(this->allas.end(),
                           lastPart.a.begin(),
                           lastPart.a.end());

        this->alljas.insert(this->alljas.end(),
                            lastPart.ja.begin(),
                            lastPart.ja.end());

        this->allias.insert(this->allias.end(),
                            lastPart.ia.begin(),
                            lastPart.ia.end());

    }


    //copy positions
    this->extendPosition(this->iapositions, this->allias.size(), repl_fact);
    this->extendPosition(this->japositions, this->alljas.size(), repl_fact);

    this->replicateVector<double>(this->allas, this->allas.size(), repl_fact);
    this->replicateVector<int>(this->allias, this->allias.size(), repl_fact);
    this->replicateVector<int>(this->alljas, this->alljas.size(), repl_fact);
    this->replicateVector<sizepair>(this->sizes, this->sizes.size(), repl_fact);
    this->replicateVector<int>(this->jaelems, this->jaelems.size(), repl_fact);
    this->replicateVector<int>(this->iaelems, this->iaelems.size(), repl_fact);

}


int SparseMatrixToSend::processFirstIndex(int size, int parts, int rank)
{
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) * rank + (rank > numSmaller ? rank - numSmaller : 0);
}

SparseMatrix SparseMatrixToSend::scatterv(std::vector<double>& recva,
                                          std::vector<int>& recvia,
                                          std::vector<int>& recvja,
                                          int num_processes,
                                          int repl_fact,
                                          int rank)
{

    sizepair my_sizes;

    MPI::COMM_WORLD.Scatter((void*)this->sizes.data(),
                            3,
                            MPI::INT,
                            (void*)&my_sizes,
                            3,
                            MPI::INT,
                            ROOT);

    recva.resize(my_sizes.ja);
    recvia.resize(my_sizes.ia);
    recvja.resize(my_sizes.ja);

    MPI::COMM_WORLD.Scatterv((void*)this->allas.data(),
                             (const int*)this->jaelems.data(),
                             (const int*)this->japositions.data(),
                             MPI::DOUBLE,
                             (void*)recva.data(),
                             my_sizes.ja,
                             MPI::DOUBLE,
                             ROOT);

    MPI::COMM_WORLD.Scatterv((void*)this->alljas.data(),
                             (const int*)this->jaelems.data(),
                             (const int*)this->japositions.data(),
                             MPI::INT,
                             (void*)recvja.data(),
                             my_sizes.ja,
                             MPI::INT,
                             ROOT);

    MPI::COMM_WORLD.Scatterv((void*)this->allias.data(),
                             (const int*)this->iaelems.data(),
                             (const int*)this->iapositions.data(),
                             MPI::INT,
                             (void*)recvia.data(),
                             my_sizes.ia,
                             MPI::INT,
                             ROOT);

    SparseMatrix my_part(recva, recvia, recvja, my_sizes);

    return my_part;

}


int SparseMatrixToSend::elemsForProcess(int size, int parts, int rank)
{
    rank %= parts;
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank > numSmaller ? 1 : 0);
}

SparseMatrix::SparseMatrix(std::vector<double>& recva,
                           std::vector<int>& recvia,
                           std::vector<int>& recvja,
                           sizepair& sizes) : Matrix(sizes.ia - 1, sizes.width),
                                              a(recva),
                                              ia(recvia),
                                              ja(recvja) {}

void printSparseMatrix(const SparseMatrix& to_print)
{

    int currentElem = 0;

    std::cout << "HEIGHT " << to_print.height << std::endl;
    std::cout << "WIDTH " << to_print.width << std::endl;
    std::cout << "non zero elements " << to_print.a.size() << std::endl << std::endl;

    for(int i = 0; i < to_print.height; i++)
    {
        for(int j = 0; j < to_print.width; j++)
        {
            if(to_print.ia[i + 1] > currentElem && to_print.ja[currentElem] == j)
            {
                std::cout << to_print.a[currentElem];
                currentElem += 1;
            } else {
                std::cout << 0;
            }
            std::cout << "\t";
        }
        std::cout << std::endl;
    }
}