#include <stdio.h>
#include <mpi.h>

int fact(int n){
    int f=1;
    for (int i=1; i<=n;i++){
        f=f*i;
    }
    return f;
}

void handling_error(int err_code){
    int err_class, length_string;
    char err_string[256];

    MPI_Error_class(err_code, &err_class);
    MPI_Error_string( err_class, err_string, &length_string);

    fprintf(stdout, "\nerror class (%d):\n %s\n",err_class, err_string);
}

int main(int argc, char* argv[]){
    int size, rank, glob;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Errhandler_set(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
    int err_code;
    MPI_Comm invalid_comm;  //change to correct comm to remove error

    int local= fact(rank);
    printf("%d!=%d\n",rank,local);
    err_code = MPI_Scan(&local, &glob, 1,MPI_INT, MPI_SUM, invalid_comm);

    if (err_code!= MPI_SUCCESS){
        handling_error(err_code);
    }

    if (rank==size-1){
        printf("%d is sum of factorials",glob);
    }

    MPI_Finalize();
    return 0;

}