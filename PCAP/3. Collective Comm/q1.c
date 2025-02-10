#include <stdio.h>
#include <mpi.h>

int fact(int n){
    int f=1;
    for (int i=1;i<=n;i++){
        f=f*i;
    }
    return f;
}

int main(int argc, char* argv[]){
    int size,rank,sum;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int s[size+1], r;

    if (rank==0){
        printf("Enter arr values:\n");
        fflush(stdout);
        for (int i=0;i<size;i++){
            scanf("%d",&s[i]);
        }
    }

    MPI_Scatter(s,1,MPI_INT,&r,1,MPI_INT,0,MPI_COMM_WORLD);
    r=fact(r);
    MPI_Gather(&r,1,MPI_INT, s,1, MPI_INT,0,MPI_COMM_WORLD);

    if(rank==0){
        fprintf(stdout, "The result gathered in the root:\n");
        fflush(stdout);
        for(int i=0;i<size;i++){
            fprintf(stdout,"%d ",s[i]);
            fflush(stdout);
            sum += s[i];
        }
        fprintf(stdout,"\nThe final sum: %d\n",sum);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}