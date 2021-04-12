#include <ctf.hpp>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>    // std::find


using namespace CTF;
using namespace CTF_int;


void writeStatistics(int rank) {
  ifstream stat("/proc/self/status");
  std::string line;
  if (rank==0){
    while (std::getline(stat, line)) {
      if (line.find("VmHWM") != string::npos) std::cout << line << std::endl;
    }
  }
}

void doSwap(
  std::vector<double> &data, World &dw, int rank,
  std::vector<int> processorPair,
  std::vector<double> &buff
)
{
  MPI_Request req; 
  int Np(data.size());
  for ( size_t i(0); i < processorPair.size(); i++){
    if ( rank == i){
      int j(processorPair[i]);
      // find the sender for the receiver rank
      std::vector<int>::iterator it;
      it = std::find(processorPair.begin(), processorPair.end(), rank); 
      int k = std::distance(processorPair.begin(), it);

      MPI_Isend(&data[0], Np, MPI_DOUBLE, j, 0, dw.comm, &req);
      //MPI_Send(&data[0], Np, MPI_DOUBLE, j, 0, dw.comm);
      MPI_Irecv(&buff[0], Np, MPI_DOUBLE, k, 0, dw.comm, &req);
      //MPI_Status status;
      //MPI_Recv(&buff[0], Np, MPI_DOUBLE, k, 0, dw.comm, &status);
    }
  }
  MPI_Barrier(dw.comm);//  is this necessary?
  double tstart(MPI_Wtime());
//  std::copy(buff.begin(), buff.begin()+data.size(), data.begin());
  memcpy(data.data(), buff.data(), Np*sizeof(double));
  MPI_Barrier(dw.comm);
  double tend(MPI_Wtime());
  double elapsed = tend - tstart;
  if (rank == 0) std::cout << "Copy in swap Time:        " << elapsed << " s\n";

}


int main(int argc, char ** argv){
  int np;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  World dw(MPI_COMM_WORLD, argc, argv);

  if (argc != 3) { std::cout << "give me the No and Nv\n"; exit(1);}
  int64_t No(std::atoi(argv[1]));
  int64_t Nv(std::atoi(argv[2]));


  // We can adjust any pairing between two ranks
  // just start with a very simple one: i <-> i+1
  std::vector<int> processorPair(np);
  for (size_t i(0); i < processorPair.size(); i++){
    int j = i+1;
    if ( j == np) j=0;
    processorPair[i] = j;
  }


  size_t Noov(No*No*Nv);
  size_t Nov(No*Nv);
  if (rank == 0) std::cout << "Number of elements: " << Noov;
  if (rank == 0) std::cout << " -> " << Noov*8./1024./1024/1024 << " GB\n";

  if (rank == 0) std::cout << "Number of elements: " << Nov;
  if (rank == 0) std::cout << " -> " << Nov*8./1024./1024/1024 << " GB\n";
 
  MPI_Barrier(MPI_COMM_WORLD);

  // construct the dummy data
  std::vector<double> buff(Noov);
  std::vector<double> dataTphh(Noov);
  std::vector<double> dataVppph(Nov);
 
 for (size_t i(0); i < dataTphh.size(); i++){
    dataTphh[i] = double(rank)*100.0 + i;
  }
 for (size_t i(0); i < dataVppph.size(); i++){
    dataVppph[i] = double(rank)*100.0 + i;
  }

/* 
  for(int j = 0; j < np; j++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (j == rank) {
      for (auto a: dataVppph){  std::cout << rank << ": " << a << std::endl;  } 
    }
  }
*/

  MPI_Barrier(MPI_COMM_WORLD);


// Every subworld consits of one core
  int child_size = 1;
  int color = rank / child_size;
  int crank = rank % child_size;
  int child_rank;
  MPI_Comm child_comm;

  MPI_Comm_split(dw.comm, color, crank, &child_comm);
  MPI_Comm_rank(child_comm, &child_rank);
  MPI_Comm_size(child_comm, &child_size);
  World child_world(child_comm);


  int syms[] = { NS, NS, NS, NS};
  int vvoo[] = { Nv, Nv, No, No};
  int voo[] = { Nv, No, No};
  int vo[]  = { Nv, No};
  int ooo[] = { No, No, No};
  auto T(new CTF::Tensor<>(3, voo, syms, child_world, "T"));
  auto V(new CTF::Tensor<>(2, vo, syms, child_world, "V"));
  auto W(new CTF::Tensor<>(3, ooo, syms, child_world, "W"));



  int tensor::set_data_pointer
      ( int64_t num_pair
      , double * data
      ) {

    ASSERT(num_pair != this->size);
    if (this->is_sparse) return ERROR;

    delete[] this->data;
    this->data = (char*)data;

    return SUCCESS;
  }



  T->data = (char*)dataTphh.data();
  //T->write_raw(dataTphh.size(), dataTphh.data());
  V->write_raw(dataVppph.size(), dataVppph.data());


  MPI_Barrier(MPI_COMM_WORLD);

  (*W)["ijk"] = (*T)["eij"] * (*V)["ek"];

 
  double tstart(MPI_Wtime());
  doSwap(dataTphh, dw, rank, processorPair, buff);
  MPI_Barrier(MPI_COMM_WORLD);
  double tend(MPI_Wtime());
  double elapsed(tend-tstart);
  double total(elapsed);
  if (rank == 0) std::cout << "1swap Time:        " << elapsed << " s\n"; 
  writeStatistics(rank);

  tstart = MPI_Wtime();
  doSwap(dataVppph, dw, rank, processorPair, buff);
  MPI_Barrier(MPI_COMM_WORLD);
  tend = MPI_Wtime();
  elapsed = tend - tstart;
  total += elapsed;

  if (rank == 0) std::cout << "2swap Time:        " << elapsed << " s\n"; 

/*
  for(int i = 0; i < np; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (i == rank) {
        for (auto a: dataVppph){  std::cout << rank << ": " << a << std::endl;  } 
      }
  }
*/


  tstart = MPI_Wtime();

  T->write_raw(dataTphh.size(), dataTphh.data());
  V->write_raw(dataVppph.size(), dataVppph.data());

  tend = MPI_Wtime();
  elapsed = tend - tstart;
  total += elapsed;
  if (rank == 0) std::cout << "Write Time:       " << elapsed << " s\n"; 
  if (rank == 0) std::cout << "WriteBandwidth:   " << (No*No*Nv+No*Nv)*2.0*8.0/1024/1024/1024/elapsed <<  " GB/s\n";
  writeStatistics(rank);
  
  tstart = MPI_Wtime();
  (*W)["ijk"] = (*T)["eij"] * (*V)["ek"];

  tend = MPI_Wtime();
  elapsed = tend - tstart;
  total += elapsed;
  if (rank == 0) std::cout << "Contract Time:    " << elapsed << " s\n"; 
 
  if (rank == 0) std::cout << "---\n";
  if (rank == 0) std::cout << "Total:            " << total << " s";
  if (rank == 0) std::cout << ", " << No*No*No*Nv*2.0/total/1e9 << " GFlops/s/core\n";
  if (rank == 0) std::cout << "Contract:  " << No*No*No*Nv*2.0/elapsed/1e9 << " GFlops/s/core\n";

  MPI_Barrier(dw.comm);//  is this necessary?

  // reference Contraction
//  auto fullA(new CTF::Tensor<>(4, vvoo, syms, dw, "fullA"));
//  auto fullC(new CTF::Tensor<>(4, vvoo, syms, dw, "fullC"));
//  
//  writeStatistics(rank);
// 
//  fullA->fill_random(0,1);
//  
//  tstart = MPI_Wtime();
//  (*fullC)["abij"] = (*fullA)["acik"] * (*fullA)["cbkj"];
//  tend = MPI_Wtime();
//  elapsed = tend-tstart; 
//  if (rank == 0) std::cout << "Ref. Contr. Time: " << elapsed*np << " s"; 
//  if (rank == 0) std::cout << ", " << No*No*No*Nv*2.0/elapsed/1e9/np << " GFlops/s/core\n";
//
//  //fullC->write_raw(dataA.size()/np, dataA.data());
//  writeStatistics(rank);
  MPI_Finalize();
  return 0;

}
