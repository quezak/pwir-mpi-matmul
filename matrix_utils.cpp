#include "matrix_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <mpi.h>

#include "densematgen.h"
#include "matrix.hpp"
#include "utils.hpp"

using namespace std;
using MPI::COMM_WORLD;


// sizes and offsets of matrix blocks that will be given initially (with c=1), indexed by rank
static vector<int> small_part_sizes, small_part_displs;
// sizes and offsets of matrix blocks after replication, indexed by rank inside "rotation group"
static vector<int> repl_part_sizes, repl_part_displs;


static int idxsForProcess(int size, int parts, int rank) {
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank >= numSmaller ? 1 : 0);
}


void initPartSizes() {
    if (!small_part_sizes.empty()) return;
    int p = Flags::procs;
    int c = Flags::repl;
    int n = Flags::size;
    ONE_DBG cerr << "p=" << p << "  c=" << c << "  n=" << n << endl;
    for (int i = 0; i < p; ++i)
        small_part_sizes.push_back(idxsForProcess(n, p, i));
    int sum = 0;
    small_part_displs.resize(p, 0);
    // Reorder the parts a bit -- it doesn't change the result (each process multiplies by
    // all block columns), but will make easier to replicate later
    for (int imod = 0; imod < p/c; ++imod) {
        for (int i = imod; i < p; i += p/c) {
            small_part_displs[i] = sum;
            sum += small_part_sizes[i];
        }
    }
    small_part_displs.push_back(sum);
    ONE_DBG cerr << " small_part_sizes: " << small_part_sizes;
    ONE_DBG cerr << "small_part_displs: " << small_part_displs;
    if (c == 1) {
        repl_part_sizes = small_part_sizes;
        repl_part_displs = small_part_displs;
    } else {
        repl_part_displs = vector<int>(small_part_displs.begin(), small_part_displs.begin() + p/c);
        repl_part_displs.push_back(sum);
        for (int i = 0; i < p/c; ++i)
            repl_part_sizes.push_back(repl_part_displs[i+1] - repl_part_displs[i]);
        ONE_DBG cerr << "  repl_part_sizes: " <<  repl_part_sizes;
        ONE_DBG cerr << " repl_part_displs: " <<  repl_part_displs;
    }
}


int partSize(bool repl, int i) {
    if (!repl) return small_part_sizes[i];
    return repl_part_sizes[i];
}


int partStart(bool repl, int i) {
    if (!repl) return small_part_displs[i];
    return repl_part_displs[i];
}


int partEnd(bool repl, int i) {
    return partStart(repl, i) + partSize(repl, i);
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
    const int n = Flags::size;
    // we need to use Gatherv, because the counts can differ if size is not divisible by p
    if (Flags::rank != ONE_WORKER_RANK) {
        COMM_WORLD.Gatherv(m.rawData(), n * partSize(false, Flags::rank), MPI::DOUBLE,
                NULL, NULL, NULL, MPI::DOUBLE,  // recv params are irrelevant for other processes
                ONE_WORKER_RANK);
    } else {
        const int p = Flags::procs;
        DenseMatrix recvM(n, n, 0, 0);
        vector<int> counts(p);
        for (int i = 0; i < p; ++i) counts[i] = n * partSize(false, i);
        vector<int> displs(Flags::procs);
        for (int i = 1; i < p; ++i) displs[i] = n * partStart(false, i);
        COMM_WORLD.Gatherv(m.rawData(), n * partSize(false, Flags::rank), MPI::DOUBLE,
                recvM.rawData(), counts.data(), displs.data(), MPI::DOUBLE,
                ONE_WORKER_RANK);
        // TODO change stream
        cerr << recvM;
    }
}


SparseMatrix splitAndScatter(SparseMatrix &m, vector<int> &nnzs) {
    int n = Flags::size;
    int p = Flags::procs;
    int r = Flags::rank;
    SparseMatrix my_part;

    if (!isMainProcess()) {
        nnzs.resize(p);
        COMM_WORLD.Bcast(nnzs.data(), p, MPI::INT, MAIN_PROCESS);
        my_part = SparseMatrix(n, partSize(false, r), 0, partStart(false, r), nnzs[r]);
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE, // ignored for non-root
                my_part.values.data(), my_part.nnz(), SparseMatrix::ELEM_TYPE,
                MAIN_PROCESS);
    } else {
        nnzs.resize(p);
        vector<int> val_displs(p);

        // Sort values by columns and find starts of each block, so we can send directly from m
        sort(m.values.begin(), m.values.end(), SparseMatrix::Elem::colOrder);
        for (int part = 0; part < p; ++part) {
            auto start_elem_it = lower_bound(m.values.begin(), m.values.end(),
                    SparseMatrix::Elem(0.0, -1, partStart(false, part)),
                    SparseMatrix::Elem::colOrder);
            auto end_elem_it = lower_bound(m.values.begin(), m.values.end(),
                    SparseMatrix::Elem(0.0, -1, partEnd(false, part)),
                    SparseMatrix::Elem::colOrder);
            nnzs[part] = end_elem_it - start_elem_it;
            val_displs[part] = start_elem_it - m.values.begin();
        }
        ONE_DBG cerr << "nnzs      : " << nnzs;
        ONE_DBG cerr << "val_displs: " << val_displs;

        // nnzs values will be needed by everyone
        COMM_WORLD.Bcast(nnzs.data(), p, MPI::INT, MAIN_PROCESS);

        // scatter the fragments
        my_part = SparseMatrix(n, partSize(false, r), 0, partStart(false, r), nnzs[r]);
        COMM_WORLD.Scatterv(m.values.data(), nnzs.data(), val_displs.data(), SparseMatrix::ELEM_TYPE,
                my_part.values.data(), my_part.nnz(), SparseMatrix::ELEM_TYPE,
                MAIN_PROCESS);
    }

    return my_part;
}


void replicateA(SparseMatrix &m, vector<int> &nnzs) {
    // Matrix division before replication: p almost-equal block columns:
    //  p[0] has first, p[p/c] has 2nd, p[2p/c] the 3rd, ...
    //  p[1] has (p/c)th, p[p/c+1] has (p/c+1)th, ...
    // So processes with equal id modulo p/c should do an allgather, then matrix will be divided
    // into p/c block columns: p[0] has 1st .. p[p/c-1] has last, p[p/c] has first again, etc.
    // When returning, nnzs should be filled for each group_comm separately.
    int c = Flags::repl;  // should be equal to repl_comm.Get_size()
    int p = Flags::procs;
    int repl_rank = Flags::repl_comm.Get_rank();
    vector<int> val_counts, val_displs;

    // Prepare count and displ vectors from subpart sizes
    val_counts.reserve(c);
    val_displs.reserve(c+1);
    val_displs.push_back(0);
    for (int i = Flags::rank % (p/c); i < p; i += p/c) {
        val_counts.push_back(nnzs[i]);
        val_displs.push_back(val_displs.back() + nnzs[i]);
    }
    ONE_DBG cerr << "val_counts: " << val_counts;
    ONE_DBG cerr << "val_displs: " << val_displs;

    // Put this process's subpart into the right part of the vectors (to do an in-place allgatherv)
    m.col_off = partStart(true, groupId());
    m.width = partSize(true, groupId());
    ONE_DBG cerr << "groupId: " << groupId() << "  new col_off: " << m.col_off << "  new width: " << m.width << endl;
    m.values.reserve(val_displs.back());  // insert needed zero elements before current ones
    m.values.insert(m.values.begin(), val_displs[repl_rank], SparseMatrix::Elem());
    ONE_DBG cerr << "after prepending " << val_displs[groupId()] << " zeroes nnz: " << m.nnz() << endl;
    m.values.resize(val_displs.back());  // append needed zero elements at the back
    ONE_DBG cerr << "resized m nnz: " << m.nnz() << endl;

    // Share the subparts
    Flags::repl_comm.Allgatherv(MPI::IN_PLACE, 0 /*ignored*/, SparseMatrix::ELEM_TYPE,
            m.values.data(), val_counts.data(), val_displs.data(), SparseMatrix::ELEM_TYPE);

    // calculate new nnzs
    vector<int> new_nnzs(p/c, 0);
    for (int i = 0; i < (int) nnzs.size(); ++i)
        new_nnzs[i % (p/c)] += nnzs[i];
    nnzs = new_nnzs;
}


DenseMatrix generateBFragment() {
    if (Flags::use_inner) {
        throw ShouldNotBeCalled("B generation for innerABC");
    } else {
        return DenseMatrix(Flags::size, partSize(false, Flags::rank),
                0, partStart(false, Flags::rank),
                generate_double, Flags::gen_seed);
    }
}


int reduceGeElems(const DenseMatrix &m_part, double bound) {
    int part_count = m_part.countGeElems(bound);
    int count = 0;
    ONE_DBG cerr << "part count: " << part_count << endl;
    COMM_WORLD.Reduce(&part_count, &count, 1, MPI::INT, MPI::SUM, ONE_WORKER_RANK);
    return count;
}
