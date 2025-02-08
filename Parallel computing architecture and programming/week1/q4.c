#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char* argv[]){
    int rank,size;
    char str[]="HelLo";
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (str[rank]>='A' && str[rank]<='Z'){
        printf("TOGGLED %c= %c\n",str[rank],str[rank]+32);
    }
    else if (str[rank]>='a' && str[rank]<='z'){
        printf("TOGGLED %c= %c\n",str[rank],str[rank]-32);
    }
    MPI_Finalize();
    return 0;
}