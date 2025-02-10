#include <stdio.h>      
#include <mpi.h>  
#include <string.h>

int main(int argc, char* argv[]){

    int size,rank,ans,local;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;

    int bsize= MPI_BSEND_OVERHEAD+ sizeof(int);
    int buffer[bsize];
    MPI_Buffer_attach(buffer,bsize);

    if (rank==0){
        for (int i=1;i<size;i++){
            MPI_Bsend(&i, 1, MPI_INT,i,0,MPI_COMM_WORLD); 
        }
    }
    else{
        MPI_Recv(&local, 1, MPI_INT,0,0,MPI_COMM_WORLD,&status);
        if (rank%2==0){
            ans= local*local;
        }
        else {
            ans= local*local*local;
        }
        printf("Process %d recieved %d\n", rank, ans);
    } 
    MPI_Buffer_detach(&buffer, &bsize);  //u have to pass address for both
    MPI_Finalize();
    return 0;
}