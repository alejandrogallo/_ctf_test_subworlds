#include <ctf.hpp>
#include <string>
#include "ctf/src/mapping/distribution.h"
#include "ctf/src/redistribution/cyclic_reshuffle.h"

using namespace CTF;
using namespace CTF_int;

struct FlopCounter {
	MPI_Comm& comm;
	std::string name;
	const double tstart;
	Flop_counter flopCounter;
	FlopCounter(std::string name_, MPI_Comm &comm_):
      comm(comm_), name(name_), tstart(MPI_Wtime()) {
		flopCounter.zero();
	}
	int64_t count() { return flopCounter.count(comm); }
	void print(int world_rank) {
		auto flopValue(count());
		int world_size;
		auto tend(MPI_Wtime());
		MPI_Comm_size(comm, &world_size);
    if (world_rank == 0){
			double gF(1024*1024*1024);
			std::cout << "FlopCounter(" << name << "):" << std::endl;
			std::cout << "\tGFlop    : " << flopValue/gF << std::endl;
			std::cout << "\tGFlops   : " << flopValue/(tend - tstart)/gF << std::endl; 
			std::cout << "\tGFlops/np: " << flopValue/(tend - tstart)/world_size/gF << std::endl; 
		}
	}
};

void writeStatistics() {
  ifstream stat("/proc/self/status");
  std::string line;
  while (std::getline(stat, line)) {
    if (line.find("VmHWM") != string::npos) std::cout << line << std::endl;
  }
}

/*
 * These functions have been taken from the ctf source
 */
void
add_to_subworld_untyped(CTF_int::tensor &L, CTF_int::tensor &R, char const *alpha, char const *beta) {
    int fw_mirror_rank, bw_mirror_rank;
    distribution * odst;
    char * sub_buffer;

    R.orient_subworld(L.wrld, bw_mirror_rank, fw_mirror_rank, odst, &sub_buffer);

    distribution idst = distribution(&L);

    cyclic_reshuffle(
      L.sym, idst, NULL, NULL, *odst, NULL, NULL, (&L.data), &sub_buffer,
       L.sr, L.wrld->cdt, 0, alpha, beta);

    MPI_Request req;
    if (fw_mirror_rank >= 0){
      //ASSERT(R != NULL);
      MPI_Irecv(R.data, odst->size, R.sr->mdtype(), fw_mirror_rank, 0, L.wrld->cdt.cm, &req);
    }

    if (bw_mirror_rank >= 0)
      MPI_Send(sub_buffer, odst->size, L.sr->mdtype(), bw_mirror_rank, 0, L.wrld->cdt.cm);
    if (fw_mirror_rank >= 0){
      MPI_Status stat;
      MPI_Wait(&req, &stat);
    }
    delete odst;
    CTF_int::cdealloc(sub_buffer);
}
void add_to_subworld(CTF::Tensor<> *L, CTF::Tensor<> *R){
  double alpha(1.0), beta(0.0);
  add_to_subworld_untyped(*L, *R, (char*)&alpha, (char*)&beta);
}
void add_to_subworld(CTF::Tensor<> *L){
  double alpha(1.0), beta(0.0);
  tensor t = tensor();
  t.sr = L->sr->clone();
  add_to_subworld_untyped(*L, t, (char*)&alpha, (char*)&beta);
  delete t.sr;
}


void doContractions(int No, int Nv, int Qpoints, World &dw){

  int syms[] = { NS, NS, NS, NS};
  int vvoo[] = { Nv, Nv, No, No};
  int entries_size = Nv*Nv*No*No;

  std::vector<CTF::Tensor<> *> T;
  for (size_t i(0); i<Qpoints; ++i){
    T.push_back(new CTF::Tensor<>(4,vvoo,syms,dw,"localT"));
  }

  int world_rank, world_size;

  MPI_Comm parent_comm = dw.comm;
  MPI_Comm_rank(parent_comm, &world_rank);
  MPI_Comm_size(parent_comm, &world_size);

	if ( world_rank == 0){ std::cout << "Number of processes: " << world_size << std::endl;  }

	// Every subworld consits of one core
  int child_size = 1;
  int color = world_rank / child_size;
  int crank = world_rank % child_size;
  int child_rank;
  MPI_Comm child_comm;

  MPI_Comm_split(parent_comm, color, crank, &child_comm);
  MPI_Comm_rank(child_comm, &child_rank);
  MPI_Comm_size(child_comm, &child_size);
  World child_world(child_comm);

  for ( int i(0); i<Qpoints; ++i){
    if ( i == world_rank){    
      double tstart(MPI_Wtime());
			FlopCounter subCounter("add_to_subworld Counter", child_comm);
      if ( world_rank == 0){ std::cout << "Start with point: " << i << std::endl;  }
      auto localT(new CTF::Tensor<>(4,vvoo,syms,child_world,"localT"));

      for (int c=0; c<world_size/child_size ; ++c){
        if(c==color){
          add_to_subworld(T[i], localT);
        }
        else{
          add_to_subworld(T[i]);
        }
      }

			//subCounter.print(world_rank);

			FlopCounter subsubCounter("Local contraction Counter", child_comm);
      double tmulti(MPI_Wtime());
      if ( world_rank == 0){ std::cout << "Added to subworld: " << tmulti-tstart << std::endl; }
      (*localT)["abij"] = (*localT)["ackj"] * (*localT)["cbik"];

			subCounter.print(world_rank);
			subsubCounter.print(world_rank);

      double tref(MPI_Wtime());
      if ( world_rank == 0){ std::cout << "Local contraction: " << tref-tmulti << std::endl; }
      // Brute force method: read whole tensor in local data container
      // write this container in a subworld tensor

      std::vector<double> data(entries_size);
      std::vector<int64_t> indices(entries_size);
      T[i]->read_all(data.data());
   
      double tread(MPI_Wtime());
      if ( world_rank == 0){ std::cout << "Read data: " << tread-tref << std::endl;  }
			//subCounter.print(world_rank);
 
      
      for (size_t it(0); it < data.size(); ++it){ indices[it] = it;}
      localT->write(data.size(), indices.data(), data.data());
      double twrite(MPI_Wtime());
      if ( world_rank == 0){ std::cout << "Wrote data: " << twrite-tread << std::endl; }
			//subCounter.print(world_rank);

      indices.clear(); data.clear();

    } // end if world_rank
  }

  FlopCounter globalCounter("Contraction with parent communicator", parent_comm);
  double tbegin(MPI_Wtime());
  for (size_t i(0); i<Qpoints; ++i){
    (*T[i])["abij"] = (*T[i])["ackj"] * (*T[i])["cbik"];
  }
  double tend(MPI_Wtime());
  if ( world_rank == 0){ std::cout << "Reference contraction(s): " << tend-tbegin << std::endl; }
  globalCounter.print(world_rank);

  MPI_Barrier(MPI_COMM_WORLD);
  // write out memory
  if (world_rank ==0){ writeStatistics(); }
}

int main(int argc, char ** argv){
  int np;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  World dw(MPI_COMM_WORLD, argc, argv);

  int No(10);
  int Nv(100);

  // We have as many qpoints as processes
  int Qpoints(np);

  doContractions(No, Nv, Qpoints, dw);
  MPI_Finalize();
  return 0;

}
