#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank > 0) {
        int msg_size;
        msg_size = world_rank + 1;

        int* buf = (int*)malloc(sizeof(int) * msg_size);
        for (int i = 0; i < msg_size; ++i) {
            buf[i] = i;
        }

        MPI_Send(buf, msg_size, MPI_INT, 0, 42, MPI_COMM_WORLD);

        fprintf(stderr, "Message sent (%d): size = %d\n", world_rank, msg_size);

        free(buf);
    } else {
        int** buf = (int**)malloc(sizeof(int*) * world_size);
        buf[0] = NULL;

        for (int i = 1; i < world_size; ++i) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 42, MPI_COMM_WORLD, &status);

            int msg_size;
            MPI_Get_count(&status, MPI_INT, &msg_size);

            buf[status.MPI_SOURCE] = (int*)malloc(sizeof(int) * msg_size);
            MPI_Recv(*(buf + status.MPI_SOURCE), msg_size, MPI_INT, status.MPI_SOURCE, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            fprintf(stderr, "Message received (%d): size = %d: \n", status.MPI_SOURCE, msg_size);
            for (int j = 0; j < msg_size; ++j) {
                fprintf(stderr, "%d ", buf[status.MPI_SOURCE][j]);
            }
            fprintf(stderr, "\n");
        }

        for (int i = 0; i < world_size; ++i) {
            free(buf[i]);
        }
        free(buf);
    }

    MPI_Finalize();
}