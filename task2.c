#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N, M, P;
    N = 13;
    M = 10;
    P = 5;
    int block_x = 4;
    int block_y = (int)ceil(1.0 * N / P);

    MPI_Datatype MX_BASE_BLOCK;
    MPI_Datatype MX_XGAP_BLOCK;
    MPI_Type_vector(1, block_x, M, MPI_INT, &MX_BASE_BLOCK);
    MPI_Type_vector(1, M % block_x, M, MPI_INT, &MX_XGAP_BLOCK);
    MPI_Type_commit(&MX_BASE_BLOCK);
    MPI_Type_commit(&MX_XGAP_BLOCK);


    int* mx = (int*)malloc(sizeof(int) * N * M);

    for (int i = 0; i < M; i += block_x) {
        if (world_rank > 0) {
            MPI_Recv(mx + (world_rank * block_y - 1) * M + i, 1, M - i < block_x ? MX_XGAP_BLOCK : MX_BASE_BLOCK, world_rank - 1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int j = world_rank * block_y; j < ((world_rank + 1) * block_y < N ? (world_rank + 1) * block_y : N); ++j) {
            for (int k = i; k < (i + block_x < M ? i + block_x : M); ++k) {
                mx[j * M + k] = (world_rank == 0 ? j : mx[(j - 1) * M + k] + 1);
            }
        }
        if (world_rank < world_size - 1) {
            MPI_Send(mx + ((world_rank + 1) * block_y - 1) * M + i, 1, M - i < block_x ? MX_XGAP_BLOCK : MX_BASE_BLOCK, world_rank + 1, 42, MPI_COMM_WORLD);
        }
    }

    if (world_rank == world_size - 1) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                fprintf(stderr, "%d ", mx[i * M + j]);
            }
            fprintf(stderr, "\n");
        }
    }

    free(mx);

    MPI_Finalize();
}