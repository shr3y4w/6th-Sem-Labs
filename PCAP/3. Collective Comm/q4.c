#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char* argv[]){

    int size,rank, size_per_process;
    char word1[100], word2[100], recv1[50], recv2[50];
    char buff[50];
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status status;

    if (rank==0){
        printf("Enter string1:\n");
        fflush(stdout);
        scanf("%s",word1);

        printf("Enter string2:\n");
        fflush(stdout);
        scanf("%s",word2);

        size_per_process = strlen(word1)/size;
    }

    MPI_Bcast(&size_per_process,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(word1,size_per_process,MPI_CHAR,recv1,size_per_process,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(word2,size_per_process,MPI_CHAR,recv2,size_per_process,MPI_INT,0,MPI_COMM_WORLD);
    char ans[100];
    int j=0,k=0;

    for (int i=0;i<2*size_per_process; i++){
        if (i%2==0){
            ans[i] = recv1[j];
            j++;
        }
        else{
            ans[i] = recv2[k];
            k++;
        }
    }
    MPI_Gather(ans,2*size_per_process,MPI_CHAR, buff,2*size_per_process, MPI_CHAR,0,MPI_COMM_WORLD);

    if(rank==0){
        printf("Final word: %s\n",buff);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}