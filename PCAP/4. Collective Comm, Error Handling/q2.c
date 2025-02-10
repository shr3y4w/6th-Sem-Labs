#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size, element, count = 0, global_count = 0;
    int matrix[3][3], local_matrix[3];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter a 3x3 matrix:\n");
        fflush(stdout);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Enter the element to search: ");
        fflush(stdout);
        scanf("%d", &element);
    }

    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(matrix, 3, MPI_INT, local_matrix, 3, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 3; i++) {
        if (local_matrix[i] == element) {
            count++;
        }
    }

    MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Element %d occurs %d times in the matrix.\n", element, global_count);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
