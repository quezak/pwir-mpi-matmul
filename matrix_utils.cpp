#include "matrix_utils.hpp"
#include <fstream>
#include <mpi.h>

#include "densematgen.h"
#include "matrix.hpp"
#include "utils.hpp"

using namespace std;
using MPI::COMM_WORLD;


int firstIdxForProcess(int size, int parts, int rank) {
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) * rank + (rank > numSmaller ? rank - numSmaller : 0);
}


int idxsForProcess(int size, int parts, int rank) {
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank >= numSmaller ? 1 : 0);
}


int idxsForProcesses(int size, int parts, int start, int end) {
    return firstIdxForProcess(size, parts, end) - firstIdxForProcess(size, parts, start);
}


int maxIdxsForProcess(int size, int parts) {
    return idxsForProcess(size, parts, parts-1);
}



bool readSparseMatrix(const string &filename, SparseMatrix &matrix) {
    ifstream input(filename);
    if (!input.is_open()) {
        cerr << "could not open file: " << filename << endl;
        return false;
    }
    input >> matrix;
    return true;
}


void gatherAndShow(DenseMatrix &m) {
    return gatherAndShow(m, Flags::procs, COMM_WORLD); 
}


void gatherAndShow(DenseMatrix &m, int parts, MPI::Intracomm &comm) {
    // we need to use Gatherv, because the counts can differ if size is not divisible by p
    if (!isMainProcess()) {
        comm.Gatherv(m.rawData(), m.elems(), MPI::DOUBLE,
                NULL, NULL, NULL, MPI::DOUBLE,  // recv params are irrelevant for other processes
                MAIN_PROCESS);
    } else {
        const int p = Flags::procs;
        const int n = Flags::size;
        DenseMatrix recvM(n, n);
        vector<int> counts(parts);
        int part_size = p / parts;
        for (int i = 0; i < parts; ++i)
            counts[i] = n * idxsForProcesses(n, p, i * part_size, (i+1) * part_size);
        vector<int> displs(Flags::procs);
        displs[0] = 0;
        for (int i = 1; i < parts; ++i)
            displs[i] = displs[i-1] + counts[i-1];
        comm.Gatherv(m.rawData(), m.elems(), MPI::DOUBLE,
                recvM.rawData(), counts.data(), displs.data(), MPI::DOUBLE,
                MAIN_PROCESS);
        // TODO change stream
        cerr << recvM;
    }
}


SparseMatrix splitAndScatter(const SparseMatrix &m, vector<int> &nnzs) {
    int partNnz = 0;
    int n = Flags::size;
    int p = Flags::procs;
    int c = Flags::repl;
    vector<double> a_v;
    vector<int> ij_v;

    if (!isMainProcess()) {
        nnzs.resize(p);
        COMM_WORLD.Bcast(nnzs.data(), p, MPI::INT, MAIN_PROCESS);
        partNnz = nnzs[Flags::rank];
        a_v.resize(partNnz);
        ij_v.resize(partNnz + n + 1);
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE,
                a_v.data(), partNnz, MPI::DOUBLE,
                MAIN_PROCESS);
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE,
                ij_v.data(), partNnz + n + 1, MPI::DOUBLE,
                MAIN_PROCESS);
    } else {
        vector<double> all_a_v;
        vector<int> all_ij_v;
        vector<int> a_displs, ij_counts, ij_displs;
        nnzs.clear();  // use nnzs instead of a_counts as it will be needed by everyone
        all_a_v.reserve(m.nnz);
        all_ij_v.reserve(m.nnz + p * (m.height + 1));
        nnzs.reserve(p);
        a_displs.reserve(p);
        ij_counts.reserve(p);
        ij_displs.reserve(p);
        // Reorder the parts a bit -- it doesn't change the result (each process multiplies by
        // all block columns), but will make easier to replicate later
        for (int imod = 0; imod < p/c; ++imod) {
            for (int i = imod; i < p; i += p/c) {
                int start = firstIdxForProcess(n, p, i);
                int end = start + idxsForProcess(n, p, i);
                SparseMatrix partM = m.getColBlock(start, end);
                partM.appendToVectors(all_a_v, nnzs, a_displs,
                        all_ij_v, ij_counts, ij_displs);
            }
        }
        COMM_WORLD.Bcast(nnzs.data(), p, MPI::INT, MAIN_PROCESS);
        partNnz = nnzs[Flags::rank];
        //DBG cerr << "all a size: " << all_a_v.size() << endl;
        DBG cerr << "nnzs:  " << nnzs;
        //DBG cerr << "a_displs:  " << a_displs;
        //DBG cerr << "all ij size: " << all_ij_v.size() << endl;
        //DBG cerr << "ij_counts:  " << ij_counts;
        //DBG cerr << "ij_displs:  " << ij_displs;
        a_v.resize(partNnz);
        ij_v.resize(partNnz + n + 1);
        COMM_WORLD.Scatterv(all_a_v.data(), nnzs.data(), a_displs.data(), MPI::DOUBLE,
                a_v.data(), partNnz, MPI::DOUBLE,
                MAIN_PROCESS);
        COMM_WORLD.Scatterv(all_ij_v.data(), ij_counts.data(), ij_displs.data(), MPI::INT,
                ij_v.data(), partNnz + n + 1, MPI::INT,
                MAIN_PROCESS);
    }

    int width = idxsForProcess(n, p, Flags::rank);
    return SparseMatrix(n, width, partNnz, a_v.begin(), ij_v.begin());
}


SparseMatrix replicateA(const SparseMatrix &m, vector<int> &nnzs) {
    // Matrix division before replication: p almost-equal block columns:
    //  p[0] has first, p[p/c] has 2nd, p[2p/c] the 3rd, ...
    //  p[1] has (p/c)th, p[p/c+1] has (p/c+1)th, ...
    // So processes with equal id modulo p/c should do an allgather, then matrix will be divided
    // into p/c block columns: p[0] has 1st .. p[p/c-1] has last, p[p/c] has first again, etc.
    // When returning, nnzs should be filled for each group_comm separately.
    int c = Flags::repl;  // should be equal to repl_comm.Get_size()
    int p = Flags::procs;
    int n = Flags::size;
    int repl_rank = Flags::repl_comm.Get_rank();  // should be in [0..c)
    vector<double> a_v;
    vector<int> ij_v;
    vector<int> a_displs, ij_counts, ij_displs, subnnzs;
    // Prepare vectors: subnnzs is used for a_counts inside the one p/c part.
    subnnzs.reserve(c);
    a_displs.reserve(c);
    ij_counts.reserve(c+1);
    ij_displs.reserve(c+1);
    int sumnnz = 0;
    int sumijcount = 0;
    for (int i = Flags::rank % p/c; i < p; i += p/c) {
        subnnzs.push_back(nnzs[i]);
        a_displs.push_back(sumnnz);
        ij_counts.push_back(nnzs[i] + n + 1);
        ij_displs.push_back(sumijcount);
        sumnnz += subnnzs.back();
        sumijcount += ij_counts.back();
    }
    a_displs.push_back(sumnnz);  // additional guards that will be helpful
    ij_displs.push_back(sumijcount);
    a_v.reserve(sumnnz);
    ij_v.reserve(sumnnz + c * (m.height + 1));
    // Put this process's subpart into the right part of the vectors (to do an in-place allgatherv
    a_v.insert(a_v.end(), a_displs[repl_rank], 0.0);
    ij_v.insert(ij_v.end(), ij_displs[repl_rank], 0.0);
    m.appendToVectors(a_v, ij_v);
    a_v.insert(a_v.end(), a_displs[c] - a_displs[repl_rank+1], 0.0);
    ij_v.insert(ij_v.end(), ij_displs[c] - ij_displs[repl_rank+1], 0.0);
    DBG cerr << "a_v size: " << a_v.size() << "  ij_v size: " << ij_v.size()
        << "  sumnnz: " << sumnnz << "  sumijcount: " << sumijcount << endl;
    // Share the subparts
    Flags::repl_comm.Allgatherv(MPI::IN_PLACE, 0 /*ignored*/, MPI::DOUBLE,
            a_v.data(), subnnzs.data(), a_displs.data(), MPI::DOUBLE);
    Flags::repl_comm.Allgatherv(MPI::IN_PLACE, 0 /*ignored*/, MPI::INT,
            ij_v.data(), ij_counts.data(), ij_displs.data(), MPI::INT);
    // Adjust the vector data so a single submatrix can be constructed:
    //  sum the ia values, offset the ja values
    vector<int> new_ij_v;
    new_ij_v.reserve(a_v.size() + n + 1);
    new_ij_v.resize(n+1, 0);
    for (int part = 0; part < c; ++part) {
        for (int i = 0; i < n+1; ++i)
            new_ij_v[i] += ij_v[i + ij_displs[part]];
        int offset = idxsForProcesses(n, p, c * groupId(), c * groupId() + part);
        for (int i = ij_displs[part] + n + 1; i < ij_displs[part+1]; ++i)
            new_ij_v.push_back(ij_v[i] + offset);
    }
    DBG cerr << "new_ij_v size: " << new_ij_v.size() << "  a_v_size+n+1: " << a_v.size() + n + 1
        << endl;
    // calculate new nnzs
    vector<int> new_nnzs(p/c, 0);
    for (int i = 0; i < (int) nnzs.size(); ++i)
        new_nnzs[i % (p/c)] += nnzs[i];
    nnzs = new_nnzs;
    // Finally, return the replicated part of A
    int width = idxsForProcesses(n, p, groupId() * c, (groupId() + 1) * c);
    return SparseMatrix(n, width, nnzs[groupId()], a_v.begin(), ij_v.begin());
}


DenseMatrix generateBFragment() {
    if (Flags::use_inner) {
        throw ShouldNotBeCalled("B generation for innerABC");
    } else {
        return DenseMatrix(Flags::size, idxsForProcess(Flags::size, Flags::procs, Flags::rank),
                generate_double, Flags::gen_seed,
                0, firstIdxForProcess(Flags::size, Flags::procs, Flags::rank));
    }
}


