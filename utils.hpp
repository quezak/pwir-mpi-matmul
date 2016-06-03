#ifndef UTILS_HPP
#define UTILS_HPP

#include <iomanip>
#include <string>
#include <exception>
#include <mpi.h>

using std::string;

#ifndef DEBUG
#define DEBUG 0
#endif

#define DBG_ID cerr << "  [" << Flags::rank << " " << std::setw(20) << std::left << __FUNCTION__ << std::right << "] "
#define IMHERE cerr << __FILE__ << ":" << __LINE__ << endl
#define DBG if (DEBUG) DBG_ID; if (DEBUG)
#define ONE_WORKER if (Flags::rank == ONE_WORKER_RANK)
#define ONE_DBG if (DEBUG && Flags::rank == ONE_WORKER_RANK) DBG_ID; if (DEBUG && Flags::rank == ONE_WORKER_RANK)
extern int ONE_WORKER_RANK;


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
    ROTATE_SPARSE_BLOCK_COL
};

// id of the "rotation group" in which the process is (which group_comm does it use)
int groupId();
int groupId(int pid);

// id of the "replication group" in which the process is (which repl_comm does it use)
int replId();
int replId(int pid);


#endif  // UTILS_HPP
