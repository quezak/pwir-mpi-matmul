#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP
#include <string>
#include <exception>

using std::string;


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


/** Return the index of first row/column owned by rank-th process when dividing matrix of a given
 * size into almost-equal parts. If rank > parts, it is taken modulo parts. */
int firstIdxForProcess(int size, int parts, int rank);


/** Return number of elements owned by rank-th process when dividing matrix of a given
 * size into almost-equal parts. If rank > parts, it is taken modulo parts. */
int elemsForProcess(int size, int parts, int rank);

#endif  // MATRIX_UTILS_HPP
