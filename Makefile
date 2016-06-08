CC       = mpicc
CXX      = mpic++
#ALLFLAGS = -Wall -g -DDEBUG=1
ALLFLAGS = -Wall -O3
CFLAGS   = $(ALLFLAGS) -std=c99
CXXFLAGS = $(ALLFLAGS) -std=c++11
LDFLAGS  = $(ALLFLAGS) -std=c++11
MATMUL   = matrixmul
DEPS     = densematgen.o matrix_utils.o matrix.o utils.o multiplicator.o
DEPFILE  = .dependencies

LATEX    = pdflatex
LTXFLAGS = -shell-escape
REPORT   = report.pdf

all: binary $(REPORT)

.PHONY: binary
binary:
	$(MAKE) $(MAKEOPTS) $(DEPFILE)
	$(MAKE) $(MAKEOPTS) $(MATMUL)

$(DEPFILE): *.cpp *.hpp *.c *.h
	$(CXX) -MM *.cpp *.c > $@

include $(DEPFILE)

$(MATMUL): $(MATMUL).o $(DEPS)
	$(CXX) $(LDFLAGS) $^ -o $@

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp %.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.pdf: %.tex
	$(LATEX) $(LTXFLAGS) $<
	$(LATEX) $(LTXFLAGS) $<

clean:
	rm -f *.o *core *~ *.out *.err $(MATMUL) *.{log,aux,toc,pdf}
