CC       = mpicc
CXX      = mpicxx
CFLAGS   = -Wall -c --std=c99 -O3
CXXFLAGS = -Wall -c --std=c++11 -O3
CLDFLAGS = -Wall -O3 --std=c99
LDFLAGS  = -Wall -O3 --std=c++11
ALL      = matrixmul
MATGENFILE = densematgen.o

all: $(ALL)

$(ALL): %: %.o $(MATGENFILE)
	$(CXX) $(LDFLAGS) $^ -o $@

%.o: %.c matgen.h Makefile
	$(CC) $(CFLAGS) $@ $<

%.o: %.cpp matgen.h
	$(CXX) $(CXXFLAGS) $@ $<

clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
