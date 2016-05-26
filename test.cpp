#include <cassert>
#include <fstream>
#include <iostream>

#include "matrix_utils.hpp"

void test_sparse_wikipedia(){
    SparseMatrix sm;
    std::ifstream file("smalltests/sparse", std::ios::in);
    file >> sm;
    assert(sm.get(0,1) == 0);
    assert(sm.get(1,0) == 5);
    assert(sm.get(0,0) == 0);
    assert(sm.get(2,2) == 3);
    assert(sm.get(3,1) == 6);
    assert(sm.get(3,2) == 0);

    assert(sm.getRowBlock(1,3).get(0,0) == 5);
    assert(sm.getRowBlock(1,3).get(0,1) == 8);
    assert(sm.getRowBlock(1,3).get(0,2) == 0);
    assert(sm.getRowBlock(1,3).get(0,3) == 0);
    assert(sm.getRowBlock(1,3).get(1,0) == 0);
    assert(sm.getRowBlock(1,3).get(1,2) == 3);

    assert(sm.getColBlock(1,3).ia[2] == 1);
    assert(sm.getColBlock(1,3).ia[3] == 2);
    assert(sm.getColBlock(1,3).ia[4] == 3);

    assert(sm.getColBlock(1,3).get(3,0) == 6);
    assert(sm.getColBlock(1,3).getRowBlock(1,3).get(1,1) == 3);

    assert(sm.getColBlock(3,4).get(0,0) == 0);
    assert(sm.getRowBlock(3,4).get(0,0) == 0);
}

int main(){
    test_sparse_wikipedia();
}