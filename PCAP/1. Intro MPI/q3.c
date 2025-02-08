#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]){
    int rank,size,a,b;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    a=10,b=5;

    if (rank==0){
        printf("%d+%d= %d\n",a,b,a+b);
    }
    else if (rank==1){
        printf("%d-%d= %d\n",a,b,a-b);
    }
    else if (rank==2){
        printf("%dx%d= %d\n",a,b,a*b);
    }
    else if (rank==3){
        printf("%d/%d= %d\n",a,b,a/b);
    }

    MPI_Finalize();
    return 0;
}