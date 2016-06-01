#include "utils.hpp"

#include <cstdlib>
#include <getopt.h>
#include <iostream>

using namespace std;


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


bool Flags::parseArgv(int argc, char **argv) {
    if (rank == NOT_SET) {
        cerr << "error: rank not set" << endl;
        return false;
    }
    int option = -1;
    while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
        switch (option) {
            case 'v': 
                show_results = true; 
                break;
            case 'i':
                use_inner = true;
                break;
            case 'f': 
                if (isMainProcess()) {
                    sparse_filename = string(optarg);
                }
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
            default:
                cerr << "error parsing argument " << option << endl;
                return false;
        }
    }
    if (gen_seed == NOT_SET) {
        cerr << "error: missing seed" << endl;
        return false;
    }
    if (isMainProcess() && sparse_filename == "") {
        cerr << "error: missing sparse matrix filename" << endl;
        return false;
    }
    return true;
}


bool isMainProcess() {
    return Flags::rank == MAIN_PROCESS;
}


const int MAIN_PROCESS = 0;


int groupId() {
    return Flags::rank / Flags::repl;
}


bool isMainGroup() {
    return groupId() == MAIN_PROCESS / Flags::repl;
}


