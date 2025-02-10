#include <stdio.h>
#include <mpi.h>

int avg(int n[],int size){
    int sm=0;
    for (int i=0;i<size;i++){
        sm+= n[i];
    }
    return sm/size;
}

int main(int argc, char* argv[]){
    int size,rank,av;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status status;
    int s[100], r[100], M, ans[size+1], total_size;

    if (rank==0){
        printf("Enter M:\n");
        fflush(stdout);
        scanf("%d",&M);
    }
    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);

    // if (rank!=0){
    //     MPI_Recv(&M,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
    // }
    // theres no need for this, since its broadcasted

    total_size= M*size;  //shouldnt do 2d array, its imp to make it a flattened array. scatter ensures m values are taken per process
    if (rank==0){
        printf("Enter %d values for all processes:\n",M);
        fflush(stdout);
        for (int i=0;i<total_size;i++){
            scanf("%d",&s[i]);
        }
    }
    MPI_Scatter(s,M,MPI_INT,r,M,MPI_INT,0,MPI_COMM_WORLD);
    av=avg(r,M);
    MPI_Gather(&av,1,MPI_INT, ans,1, MPI_INT,0,MPI_COMM_WORLD);

    int my_sum=0;
    if(rank==0){
        printf("The averages per process:\n");
        fflush(stdout);
        for(int i=0;i<size;i++){
            fprintf(stdout,"%d ",ans[i]);
            fflush(stdout);
            my_sum+=ans[i];
        }
    }
    if (rank==0){
        printf("\nThe total average: %d\n",my_sum/size);
    }
    MPI_Finalize();
    return 0;
}