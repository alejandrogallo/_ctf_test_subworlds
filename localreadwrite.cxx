#include <ctf.hpp>

using namespace CTF;



void energyDenominator(int No, int Nv, int rank, World &dw) {
  // prepare data:
  int o[] = {No};
  int v[] = {Nv};
  int sym[] = { NS};
  Tensor<> epsi(1, o, sym, dw, "epsi");
  Tensor<> epsa(1, v, sym, dw, "epsi");
  std::vector<double> occupiedEnergies(No);
  std::vector<double> virtualEnergies(Nv);
  std::vector<int64_t> occupiedIndices(No);
  std::vector<int64_t> virtualIndices(Nv);

  for ( size_t it(0); it < occupiedEnergies.size(); ++it){
    occupiedIndices[it] = it;
    occupiedEnergies[it] = -100. + (double)it;
  }
  for ( size_t it(0); it < virtualEnergies.size(); ++it){
    virtualIndices[it] = it;
    virtualEnergies[it] = (double)it;
  }

  epsi.write(occupiedIndices.size(), occupiedIndices.data(), occupiedEnergies.data());
  epsa.write(virtualIndices.size(), virtualIndices.data(), virtualEnergies.data());


  int vvvooo[] = { Nv, Nv, Nv, No, No, No };
  int   syms[] = { NS, NS, NS, NS, NS, NS };
  Tensor<> E(6, vvvooo, syms, dw, "E");

// naive construction would be a very inefficient task

//  E["abcijk"]  = (*epsi)["i"];
//  E["abcijk"] += (*epsi)["j"];
//  E["abcijk"] += (*epsi)["k"];
//  E["abcijk"] -= (*epsa)["a"];
//  E["abcijk"] -= (*epsa)["b"];
//  E["abcijk"] -= (*epsa)["c"];

    double tstart(MPI_Wtime());

    std::vector<int64_t> indices;
    int64_t *globalIndices;
    double *data;
    int64_t localIndicesSize;

    E.read_local(&localIndicesSize, &globalIndices, &data);

    double tread(MPI_Wtime());
    if(rank == 0){
      std::cout << "READ: " << tread-tstart << std::endl;
    }

    //E.get_local_data(&localIndicesSize, &globalIndices, &data);

    double *epsidata;
    double *epsadata;

    std::vector<double> nominator(localIndicesSize);
    epsi.read_all((int64_t *)&No, &epsidata);
    epsa.read_all((int64_t *)&Nv, &epsadata);

    for (size_t it(0); it<localIndicesSize; it++) {
      // g = a + b*Nv + c*Nv^2 + i*Nv^3 + j*Nv^3*No + k*Nv^3*No^2
      size_t g = globalIndices[it];
      double sv(0);
      size_t a = (g % Nv);
      size_t b = (g / Nv) % Nv;
      size_t c = (g / (Nv * Nv)) % Nv;
      size_t i = (g / (Nv * Nv * Nv)) % No;
      size_t j = (g / (Nv * Nv * Nv * No)) % No;
      size_t k = (g / (Nv * Nv * Nv * No * No));
      sv -= epsadata[a];
      sv -= epsadata[b];
      sv -= epsadata[c];
      sv += epsidata[i];
      sv += epsidata[j];
      sv += epsidata[k];
      nominator[it] = sv;
    }

    double twork(MPI_Wtime());
    if(rank == 0){
      std::cout << "WORK: " << twork-tread << std::endl;
    }
    E.write(localIndicesSize, globalIndices, nominator.data());

    double twrite(MPI_Wtime());

    if(rank == 0){
      std::cout << "WRITE: " << twrite-twork << std::endl;
    }

    // profiling of the reference implementation
    E["abcijk"]  = epsi["i"];
    E["abcijk"] += epsi["j"];
    E["abcijk"] += epsi["k"];
    E["abcijk"] -= epsa["a"];
    E["abcijk"] -= epsa["b"];
    E["abcijk"] -= epsa["c"];

    double tref(MPI_Wtime());
    if(rank == 0){
      std::cout << "Reference: " << tref-twrite << std::endl;
    }
}


int main(int argc, char ** argv){
  int rank, np;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  World dw(MPI_COMM_WORLD, argc, argv);
  int No, Nv;
  if (argc == 3) {
   No = (std::atoi(argv[1]));
   Nv = (std::atoi(argv[2]));
  } else {
    exit(1);
  }

  energyDenominator(No, Nv, rank, dw);


  MPI_Finalize();
  return 0;

}
