#include <cassert>
#include <fstream>
#include <iostream>

#include "matrix_utils.hpp"

void test_sparse_wikipedia(){
    SparseMatrix sm{4,4};
    std::ifstream file("smalltests/sparse", std::ios::in);
    file >> sm;
    assert(sm.at(0,1) == 0);
    assert(sm.at(1,0) == 5);
    assert(sm.at(0,0) == 0);
    assert(sm.at(2,2) == 3);
    assert(sm.at(3,1) == 6);
    assert(sm.at(3,2) == 0);
}

int main(){
    test_sparse_wikipedia();
}