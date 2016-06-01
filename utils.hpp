#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <exception>
#include <mpi.h>

using std::string;

#ifndef DEBUG
#define DEBUG 0
#endif

#define DBG if (DEBUG) cerr << "[" << Flags::rank << "] "; if (DEBUG)
#define IMHIR cerr << __FILE__ << ":" << __LINE__ << endl

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
    ShouldNotBeCalled(const string &s): Exception(s) {}
};


class RuntimeError: public Exception {
public:
    RuntimeError(const string &s): Exception(s) {}
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
    static MPI::Intracomm group_comm;  // processes which have different data (and together have the whole matrix)
    static MPI::Intracomm repl_comm;  // processes which have the same replicated data

    /// Process all the commandline options, return true if successful
    static bool parseArgv(int argc, char **argv);
};

bool isMainProcess();
extern const int MAIN_PROCESS;


enum MSG_TAGS {
    ROTATE_SPARSE_A,
    ROTATE_SPARSE_IJ
};

// id of the replication group in which the process is
int groupId();

bool isMainGroup();


#endif  // UTILS_HPP
