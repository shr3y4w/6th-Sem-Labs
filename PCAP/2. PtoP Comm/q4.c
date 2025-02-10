#include <stdio.h>      
#include <mpi.h>  
#include <string.h>

int main(int argc, char* argv[]){

    int size,rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;
    int n,local,sending;

    if (rank==0){
        n=1;
        MPI_Ssend(&n, 1, MPI_INT,1,0,MPI_COMM_WORLD); 
        MPI_Recv(&local, 1, MPI_INT, size-1 ,0, MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv(&local, 1, MPI_INT, rank-1 ,0, MPI_COMM_WORLD, &status);
        sending=local+1;

        MPI_Ssend(&sending, 1, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);   //CYCLIC
        
    } 
    printf("process %d recieved %d",rank,local);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}