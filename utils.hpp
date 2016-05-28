#ifndef UTILS_HPP
#define UTILS_HPP

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


/// Static class to hold all the global settings
class Flags {
public:
    static constexpr int NOT_SET = -1;
    static int procs;
    static int rank;
    static bool show_results;
    static bool use_inner;
    static int gen_seed;
    static int repl;
    static bool count_ge;
    static double ge_element;
    static int exponent;
    static string sparse_filename;
    static int size;

    /// Process all the commandline options, return true if successful
    static bool parseArgv(int argc, char **argv);
};

bool isMainProcess();

#endif  // UTILS_HPP
