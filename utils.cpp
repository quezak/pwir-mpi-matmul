#include "utils.hpp"

#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <mpi.h>

#include "matrix_utils.hpp"

using MPI::COMM_WORLD;
using namespace std;


int ONE_WORKER_RANK = 0;

int Flags::procs = 1;
int Flags::rank = NOT_SET;
bool Flags::show_results = false;
bool Flags::use_inner = false;
int Flags::gen_seed = NOT_SET;
int Flags::repl = 1;
bool Flags::count_ge = false;
double Flags::ge_element = 0;
int Flags::exponent = 1;
string Flags::sparse_filename = "";
int Flags::size = NOT_SET;
MPI::Intracomm Flags::group_comm;
MPI::Intracomm Flags::repl_comm;
MPI::Intracomm Flags::team_comm;


bool Flags::parseArgv(int argc, char **argv) {
    if (rank == NOT_SET) {
        cerr << "error: rank not set" << endl;
        return false;
    }
    int option = -1;
    while ((option = getopt(argc, argv, "vis:f:c:e:g:O:")) != -1) {
        switch (option) {
            case 'v': 
                show_results = true; 
                break;
            case 'i':
                use_inner = true;
                break;
            case 'f': 
                sparse_filename = string(optarg);
                break;
            case 'c': 
                repl = atoi(optarg);
                break;
            case 's':
                gen_seed = atoi(optarg);
                break;
            case 'e': 
                exponent = atoi(optarg);
                break;
            case 'g': 
                count_ge = true; 
                ge_element = atof(optarg);
                break;
            case 'O':
                ONE_WORKER_RANK = atoi(optarg);
                break;
            default:
                cerr << "error parsing argument " << option << endl;
                return false;
        }
    }
    if (gen_seed == NOT_SET) {
        ONE_WORKER cerr << "error: missing seed" << endl;
        return false;
    }
    if (sparse_filename == "") {
        ONE_WORKER cerr << "error: missing sparse matrix filename" << endl;
        return false;
    }
    if (use_inner && (procs % (repl*repl))) {
        ONE_WORKER cerr << "error: in innerABC p should be divisible by c^2" << endl;
        return false;
    }
    if (procs % repl) {
        ONE_WORKER cerr << "error: p should be divisible by c" << endl;
        return false;
    }
    return true;
}


bool isMainProcess() {
    return Flags::rank == MAIN_PROCESS;
}


const int MAIN_PROCESS = 0;


int groupId(int pid) { return pid % (Flags::procs / Flags::repl); }
int groupId() { return groupId(Flags::rank); }


int replId(int pid) { return pid / Flags::repl; }
int replId() { return replId(Flags::rank); }


static void initCommsColA();
static void initCommsInnerABC();
void initGroupComms() {
    if (Flags::use_inner) initCommsInnerABC();
    else initCommsColA();
    if (Flags::repl > 1) {
        ONE_DBG cerr << "repl comm rank: " << Flags::repl_comm.Get_rank() 
            << "  size: " << Flags::repl_comm.Get_size() << endl;
    }
    ONE_DBG cerr << "group comm rank: " << Flags::group_comm.Get_rank() 
        << "  size: " << Flags::group_comm.Get_size() << endl;
}


static void initCommsColA() {
    if (Flags::repl > 1) {
        // Communicator to replicate data (processes will have the same part of A)
        // Unneded if no replication is done
        int repl_id = Flags::rank % (Flags::procs / Flags::repl);
        Flags::repl_comm = COMM_WORLD.Split(repl_id, Flags::rank);
    }
    // Communicator to rotate data (processes will have different parts, and together the whole A)
    // Will be just one comm if c=1
    int group_id = Flags::rank / (Flags::procs / Flags::repl);
    Flags::group_comm = COMM_WORLD.Split(group_id, Flags::rank);
}


static void initCommsInnerABC() {
    int r = Flags::rank;
    int c = Flags::repl;
    int p = Flags::procs;
    if (c > 1) {
        int team_id = innerBWhichReplGroup(r);
        Flags::team_comm = COMM_WORLD.Split(team_id, r);
        ONE_DBG cerr << "team comm rank: " << Flags::team_comm.Get_rank()
            << "  size: " << Flags::team_comm.Get_size() << endl;
        Flags::repl_comm = COMM_WORLD.Split(innerAWhichReplGroup(r), r);
    }
    // Comm to rotate data (the processes together have the whole A), each will have p/c processes,
    // and the ranks should go as in parts_order in initPartSizesInnerA(), e.g. for p=27, c=3
    // the first group comm should contain [0, 3, 6, 1, 4, 7, 2, 5, 8], in that order.
    int group_id = r / (p/c);
    int group_rank = ((r % c) * p) + r;  // the ranks don't have to be continous
    Flags::group_comm = COMM_WORLD.Split(group_id, group_rank);
}


int innerGroupId() { return innerGroupId(Flags::rank); }
int innerGroupId(int pid) {
    int p = Flags::procs;
    int c = Flags::repl;
    return ((pid / (p/c)) * c) + (pid % c);
}

