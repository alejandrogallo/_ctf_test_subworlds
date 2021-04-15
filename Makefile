CTFDIR = /home/fs71337/airmler/runs/ctftest/ctf/ctf
INCLUDE = $(CTFDIR)/include
CTF = $(CTFDIR)/lib

#TARGETS = localreadwrite multiworld newmulti
#TARGETS = newmulti mpiswap
TARGETS = newtriples

.PHONY: all

all: $(TARGETS)

%: %.cxx
	mpiicc $< \
		-std=c++11 \
		-qopenmp -qoverride-limits -DINTEL_COMPILER -O3 \
		-L${CTF} -lctf \
		-I${INCLUDE} \
		-fmax-errors=3 -pedantic \
		-mkl -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 \
		-o $@

clean:
	rm -v $(TARGETS)

include ta.mk
