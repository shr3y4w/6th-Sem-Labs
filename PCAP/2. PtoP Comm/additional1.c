#include <stdio.h>
#include <mpi.h>
#include <math.h>

int is_prime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int rank, size, num;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int arr[size];
    if (rank == 0) {
        printf("Enter %d elements: ", size);
        fflush(stdout);
        for (int i = 0; i < size; i++) {
            scanf("%d", &arr[i]);
        }
        for (int i = 1; i < size; i++) {
            MPI_Send(&arr[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        num = arr[0];
    } else {
        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    int prime = is_prime(num);
    printf("Process %d: %d is %s\n", rank, num, prime ? "prime" : "not prime");
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
