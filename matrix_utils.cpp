#include "matrix_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <mpi.h>

#include "densematgen.h"
#include "matrix.hpp"
#include "utils.hpp"

using namespace std;
using MPI::COMM_WORLD;


// sizes and offsets of matrix blocks that will be given initially (with c=1), indexed by rank
static vector<int> colA_small_counts, colA_small_displs,
    iA_small_counts, iA_small_displs,
    iB_small_counts, iB_small_displs;
// sizes and offsets of matrix blocks after replication, indexed by rank inside "rotation group"
static vector<int> colA_repl_counts, colA_repl_displs,
    iA_repl_counts, iA_repl_displs,
    iB_repl_counts, iB_repl_displs;


static int idxsForPart(int size, int parts, int rank) {
    int numSmaller = parts - (size % parts);  // number of parts that are smaller by one element
    return (size / parts) + (rank >= numSmaller ? 1 : 0);
}


static void initPartSizesColA();
static void initPartSizesInnerA();
static void initPartSizesInnerB();
void initPartSizes() {
    if (Flags::use_inner) {
        if (!iB_small_counts.empty()) return;
        initPartSizesInnerA();
        initPartSizesInnerB();
    } else {
        if (!colA_small_counts.empty()) return;
        initPartSizesColA();
    }
}


static void initPartSizesColA() {
    int p = Flags::procs;
    int c = Flags::repl;
    int n = Flags::size;
    ONE_DBG cerr << "p=" << p << "  c=" << c << "  n=" << n << endl;
    for (int i = 0; i < p; ++i)
        colA_small_counts.push_back(idxsForPart(n, p, i));
    int sum = 0;
    colA_small_displs.resize(p, 0);
    // Reorder the parts a bit -- it doesn't change the result (each process multiplies by
    // all block columns), but will make easier to replicate later
    for (int imod = 0; imod < p/c; ++imod) {
        for (int i = imod; i < p; i += p/c) {
            colA_small_displs[i] = sum;
            sum += colA_small_counts[i];
        }
    }
    colA_small_displs.push_back(sum);
    ONE_DBG cerr << "colA_small_counts: " << colA_small_counts;
    ONE_DBG cerr << "colA_small_displs: " << colA_small_displs;
    if (c == 1) {
        colA_repl_counts = colA_small_counts;
        colA_repl_displs = colA_small_displs;
    } else {
        colA_repl_displs = vector<int>(colA_small_displs.begin(), colA_small_displs.begin() + p/c);
        colA_repl_displs.push_back(sum);
        for (int i = 0; i < p/c; ++i)
            colA_repl_counts.push_back(colA_repl_displs[i+1] - colA_repl_displs[i]);
        ONE_DBG cerr << " colA_repl_counts: " <<  colA_repl_counts;
        ONE_DBG cerr << " colA_repl_displs: " <<  colA_repl_displs;
    }
}


static void initPartSizesInnerA() {
    int p = Flags::procs;
    int c = Flags::repl;
    int n = Flags::size;
    vector<int> part_order(p);
    for (int i = 0; i < p; ++i) part_order[i] = i;
    sort(part_order.begin(), part_order.end(), [](int a, int b) {
            // assign the parts first by repl group, then rank
            return (innerAWhichReplGroup(a) == innerAWhichReplGroup(b)
                    ? a < b
                    : innerAWhichReplGroup(a) < innerAWhichReplGroup(b));
            });
    // assign sizes in part_order, to get more evenly distributed sizes after replication
    for (int i = 0; i < p; ++i)
        iA_small_counts.push_back(idxsForPart(n, p, part_order[i]));
    int sum = 0;
    iA_small_displs.resize(p+1);
    for (int i = 0; i < p; ++i) {
        iA_small_displs[part_order[i]] = sum;
        sum += iA_small_counts[part_order[i]];
    }
    iA_small_displs[p] = sum;
    ONE_DBG cerr << "     part_order: " << part_order;
    ONE_DBG cerr << "iA_small_counts: " << iA_small_counts;
    ONE_DBG cerr << "iA_small_displs: " << iA_small_displs;
    if (c == 1) {
        iA_repl_counts = iA_small_counts;
        iA_repl_displs = iA_small_displs;
    } else {
        for (int i = 0; i < p/c; ++i) {
            iA_repl_counts.push_back(accumulate(iA_small_counts.begin() + (i*c),
                        iA_small_counts.begin() + ((i+1)*c), 0));
        }
        iA_repl_displs.resize(p/c+1, 0);
        partial_sum(iA_repl_counts.begin(), iA_repl_counts.end(), iA_repl_displs.begin()+1);
        ONE_DBG cerr << " iA_repl_counts: " << iA_repl_counts;
        ONE_DBG cerr << " iA_repl_displs: " << iA_repl_displs;
    }
}


static void initPartSizesInnerB() {
    int p = Flags::procs;
    int c = Flags::repl;
    int n = Flags::size;
    // assign the sizes out-of-order, to get more evenly distributed sizes after replication
    for (int imod = 0; imod < p/c; ++imod)
        for (int i = imod; i < p; i += p/c)
            iB_small_counts.push_back(idxsForPart(n, p, i));
    iB_small_displs.resize(p+1, 0);
    partial_sum(iB_small_counts.begin(), iB_small_counts.end(), iB_small_displs.begin()+1);
    ONE_DBG cerr << "iB_small_counts: " << iB_small_counts;
    ONE_DBG cerr << "iB_small_displs: " << iB_small_displs;
    if (c == 1) {
        iB_repl_counts = iB_small_counts;
        iB_repl_displs = iB_small_displs;
    } else {
        for (int i = 0; i < p/c; ++i) {
            iB_repl_counts.push_back(accumulate(iB_small_counts.begin() + (i*c),
                        iB_small_counts.begin() + ((i+1)*c), 0));
        }
        iB_repl_displs.resize(p/c+1, 0);
        partial_sum(iB_repl_counts.begin(), iB_repl_counts.end(), iB_repl_displs.begin()+1);
        ONE_DBG cerr << " iB_repl_counts: " << iB_repl_counts;
        ONE_DBG cerr << " iB_repl_displs: " << iB_repl_displs;
    }
}


int innerAWhichReplGroup(int i) {
    int p = Flags::procs;
    int c = Flags::repl;
    return ((i / c) + ((i % c) * (p/(c*c)))) % (p/c);
}


static int colAPartSize(bool repl, int i) { return repl ? colA_repl_counts[i] : colA_small_counts[i]; }
static int colAPartStart(bool repl, int i) { return repl ? colA_repl_displs[i] : colA_small_displs[i]; }
static int colAPartEnd(bool repl, int i) { return colAPartStart(repl, i) + colAPartSize(repl, i); }


static int innerAPartSize(bool repl, int i) { return repl ? iA_repl_counts[i] : iA_small_counts[i]; }
static int innerAPartStart(bool repl, int i) { return repl ? iA_repl_displs[i] : iA_small_displs[i]; }
static int innerAPartEnd(bool repl, int i) { return innerAPartStart(repl, i) + innerAPartSize(repl, i); }


int partASize(bool repl, int i) { return Flags::use_inner ? innerAPartSize(repl, i) : colAPartSize(repl,i); }
int partAStart(bool repl, int i) { return Flags::use_inner ? innerAPartStart(repl, i) : colAPartStart(repl,i); }
int partAEnd(bool repl, int i) { return Flags::use_inner ? innerAPartEnd(repl, i) : colAPartEnd(repl,i); }


int innerBPartSize(bool repl, int i) { return repl ? iB_repl_counts[i] : iB_small_counts[i]; }
int innerBPartStart(bool repl, int i) { return repl ? iB_repl_displs[i] : iB_small_displs[i]; }
int innerBPartEnd(bool repl, int i) { return innerBPartStart(repl, i) + innerBPartSize(repl, i); }


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
        COMM_WORLD.Gatherv(m.rawData(), n * partASize(false, Flags::rank), MPI::DOUBLE,
                NULL, NULL, NULL, MPI::DOUBLE,  // recv params are irrelevant for other processes
                ONE_WORKER_RANK);
    } else {
        const int p = Flags::procs;
        DenseMatrix recvM(n, n, 0, 0);
        vector<int> counts(p);
        for (int i = 0; i < p; ++i) counts[i] = n * partASize(false, i);
        vector<int> displs(Flags::procs);
        for (int i = 1; i < p; ++i) displs[i] = n * partAStart(false, i);
        COMM_WORLD.Gatherv(m.rawData(), n * partASize(false, Flags::rank), MPI::DOUBLE,
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
        if (Flags::use_inner) {  // receive a block-row for innerABC
            my_part = SparseMatrix(partASize(false, r), n, partAStart(false, r), 0, nnzs[r]);
        } else {  // receive a block-col for colA
            my_part = SparseMatrix(n, partASize(false, r), 0, partAStart(false, r), nnzs[r]);
        }
        COMM_WORLD.Scatterv(NULL, NULL, NULL, MPI::DOUBLE, // ignored for non-root
                my_part.values.data(), my_part.nnz(), SparseMatrix::ELEM_TYPE,
                MAIN_PROCESS);
    } else {
        nnzs.resize(p);
        vector<int> val_displs(p);

        // Sort values by row/col and find starts of each block, so we can send directly from m
        auto elem_comparator = (Flags::use_inner
                ? SparseMatrix::Elem::rowOrder
                : SparseMatrix::Elem::colOrder);
        sort(m.values.begin(), m.values.end(), elem_comparator);
        for (int part = 0; part < p; ++part) {
            auto start_elem = (Flags::use_inner
                    ? SparseMatrix::Elem(0.0, partAStart(false, part), -1)
                    : SparseMatrix::Elem(0.0, -1, partAStart(false, part)));
            auto end_elem = (Flags::use_inner
                    ? SparseMatrix::Elem(0.0, partAEnd(false, part), -1)
                    : SparseMatrix::Elem(0.0, -1, partAEnd(false, part)));
            auto start_elem_it = lower_bound(m.values.begin(), m.values.end(),
                    start_elem, elem_comparator);
            auto end_elem_it = lower_bound(m.values.begin(), m.values.end(),
                    end_elem, elem_comparator);
            nnzs[part] = end_elem_it - start_elem_it;
            val_displs[part] = start_elem_it - m.values.begin();
        }
        ONE_DBG cerr << "nnzs      : " << nnzs;
        ONE_DBG cerr << "val_displs: " << val_displs;

        // nnzs values will be needed by everyone
        COMM_WORLD.Bcast(nnzs.data(), p, MPI::INT, MAIN_PROCESS);

        // scatter the fragments
        if (Flags::use_inner) {  // receive a block-row for innerABC
            my_part = SparseMatrix(partASize(false, r), n, partAStart(false, r), 0, nnzs[r]);
        } else {  // receive a block-col for colA
            my_part = SparseMatrix(n, partASize(false, r), 0, partAStart(false, r), nnzs[r]);
        }
        COMM_WORLD.Scatterv(m.values.data(), nnzs.data(), val_displs.data(), SparseMatrix::ELEM_TYPE,
                my_part.values.data(), my_part.nnz(), SparseMatrix::ELEM_TYPE,
                MAIN_PROCESS);
    }

    return my_part;
}


void replicateA(SparseMatrix &m, vector<int> &nnzs) {
    if (Flags::use_inner) {
        throw ShouldNotBeCalled("A replication in innerABC");
    }
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
    m.col_off = colAPartStart(true, groupId());
    m.width = colAPartSize(true, groupId());
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
    if (!Flags::use_inner) {
        return DenseMatrix(Flags::size, partASize(false, Flags::rank),
                0, partAStart(false, Flags::rank),
                generate_double, Flags::gen_seed);
    } else {
        int c = Flags::repl;
        if (c == 1) return DenseMatrix(Flags::size, innerBPartSize(false, Flags::rank),
                0, innerBPartStart(false, Flags::rank),
                generate_double, Flags::gen_seed);
        throw ShouldNotBeCalled("B replication in innerABC");
    }
}


int reduceGeElems(const DenseMatrix &m_part, double bound) {
    int part_count = m_part.countGeElems(bound);
    int count = 0;
    ONE_DBG cerr << "part count: " << part_count << endl;
    COMM_WORLD.Reduce(&part_count, &count, 1, MPI::INT, MPI::SUM, ONE_WORKER_RANK);
    return count;
}
