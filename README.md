# pwir-mpi-matmul

This code implements two sparse-dense matrix multiplication algorithms with reduced communication:
**1.5D Blocked Column A** and **1.5D Blocked Inner ABC**, described in [a paper by P. Koanantakool et al](http://www.eecs.berkeley.edu/~penpornk/spdm3\_ipdps16.pdf). It was a final assignment on a course
in Parallel and Distributed Computing. It was written in C++11 with MPI libraries,
and tested with `openmpi-1.10.2` on my computer and `mpich2` on our student server.
A detailed report in Polish can be compiled by running `make report.pdf`.
