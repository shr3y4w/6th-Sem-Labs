// {18, 523, 301, 1234, 2, 14, 108, 150, 1928}

#include <stdio.h>
#include <mpi.h>

int rev(int num){
    int rev=0;

    while(num!=0){
        int n = num%10;
        rev = rev*10 + n;
        num = num/10;
    }
    return rev;
}

int main( int argc, char* argv[]){

    int rank,size;
    int arr[]={18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int out[9]={0};  //so that its initialized, not garbage value
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    out[rank]= rev(arr[rank]);
    printf("%d reversed= %d ", arr[rank],out[rank]);

    MPI_Finalize();
    return 0;
}