#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char* argv[]){

    int size,rank, size_per_process, loc_count=0;
    char word[100], recv[50];
    int buff[50];
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Status status;

    if (rank==0){
        printf("Enter string:\n");
        fflush(stdout);
        scanf("%s",word);

        size_per_process = strlen(word)/size;
    }
    MPI_Bcast(&size_per_process,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(word,size_per_process,MPI_CHAR,recv,size_per_process,MPI_INT,0,MPI_COMM_WORLD);
    
    for( int i=0; i<size_per_process; i++){
        if (recv[i]!='a' && recv[i]!='e' && recv[i]!='i' && recv[i]!='o' && recv[i]!='u' &&
            recv[i]!='A' && recv[i]!='E' && recv[i]!='I' && recv[i]!='O' && recv[i]!='U'){
            loc_count++;
        }
    }

    int bsum=0;
    MPI_Gather(&loc_count,1,MPI_INT, buff,1, MPI_INT,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("The count of non-vowel per process:\n");
        fflush(stdout);
        for(int i=0;i<size;i++){
            bsum++;
            fprintf(stdout,"%d ",buff[i]);
            fflush(stdout);
        }
        printf("\nTotal non-vowels: %d", bsum);
    }
    MPI_Finalize();
    return 0;
}