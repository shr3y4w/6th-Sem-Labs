#include <stdio.h>      
#include <mpi.h>  
#include <string.h>

int main(int argc, char* argv[]){
    int size,rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;
    int n, local;

    if (rank==0){
        for (int i=1;i<size;i++){
            int sending=i*10;
            MPI_Send(&sending, 1, MPI_INT,i,0,MPI_COMM_WORLD); 
        }
    }
    else{
        MPI_Recv(&local, 1, MPI_INT,0,0,MPI_COMM_WORLD,&status);
        printf("Process %d recieved %d\n", rank, local);
    }

    MPI_Finalize();
    return 0;
}