#include <stdio.h>
#include <mpi.h>
#include <math.h>

long long fib(int n){
    long long n1=0;
    long long n2=1;
    long long n3;
    
    if (n==0 || n==1){
        return n;
    }
    else {
        for (int i=2; i<n; i++){
            n3=n1+n2;
            n1=n2;
            n2=n3;
        }
        return n3;
    }
}

long long fact( int n){
    if (n==0 || n==1){
        return 1;
    }
    long long f=1;
    for (long long i=1; i<=n; i++){
        f = f*i;
    }
    return f;
}

int main(int argc, char* argv[]){
    int rank,size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank%2==0){
        printf("factorial no (%d): %lld \n", rank, fact(rank));
    }
    else {
        printf("fibonacci no (%d): %lld \n", rank, fib(rank));
    }

    MPI_Finalize();
    return 0;
}