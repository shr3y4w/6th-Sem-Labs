#include <stdio.h>      
#include <mpi.h>  
#include <string.h>

int main(int argc, char* argv[]){
    int size,rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Status status;
    char word[100], ans[100];

    if (rank==0){
        printf("Enter word:\n");
        fflush(stdout);  // imp because it gets buffered
        scanf("%s",word);
        MPI_Ssend(word, strlen(word)+1, MPI_CHAR,1,0,MPI_COMM_WORLD);   //while sending keep size as the size of string+1, but return a large no
        MPI_Recv(ans, strlen(word)+1, MPI_CHAR,1,1,MPI_COMM_WORLD,&status);
        printf("Recieved toggled ans (%s) by process %d\n",ans,rank);
    }
    else{
        MPI_Recv(word, 100, MPI_CHAR,0,0,MPI_COMM_WORLD,&status);
        printf("Recieved %s by process %d\n",word,rank);

        for (int i=0;i<strlen(word);i++){

            if (word[i]>='A' && word[i]<='Z'){
                ans[i]= word[i]+32;
            }

            else if (word[i]>='a' && word[i]<='z'){
                ans[i]= word[i]-32;
            }
        }
        MPI_Ssend(ans, strlen(word)+1, MPI_CHAR,0,1,MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}