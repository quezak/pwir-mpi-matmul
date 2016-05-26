#include <fstream>
#include <iostream>

#include "matrix_utils.hpp"

void test_sparse_wikipedia(){
    SparseMatrix sm{4,4};
    std::ifstream file("smalltests/sparse", std::ios::in);
    file >> sm;
    std::cout << sm.at(0,1) << std::endl;
    std::cout << sm.at(1,0) << std::endl;
    std::cout << sm.at(0,0) << std::endl;
}

int main(){
    test_sparse_wikipedia();
}