#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) 
{
    if (argc != 4) {
        fprintf(stderr, "Missing arguments\n");
        return 2;
    }

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int N, M, P;
    N = atoi(argv[1]);//20;//matrix height
    M = atoi(argv[2]);//15;//matrix width
    P = atoi(argv[3]);//4;//num. procs

    int BLOCKDIM_X = 5;
    int BLOCKDIM_Y = (int)ceil(1.0 * N / P);
    int HOLLOW_TOP = 1;
    int BLOCK_NUM = (int)ceil(1.0 * M / BLOCKDIM_X);//num. of blocks per strip

    //hollow region included
    int BLOCKDIM_Y__LOCAL = ((world_rank + 1) * BLOCKDIM_Y < N ? BLOCKDIM_Y : N - world_rank * BLOCKDIM_Y) + (world_rank == 0 ? 0 : HOLLOW_TOP);

    MPI_Datatype block_base_t;
    MPI_Datatype block_gap_x_t;
    MPI_Type_vector(HOLLOW_TOP, BLOCKDIM_X, M, MPI_INT, &block_base_t);
    MPI_Type_vector(HOLLOW_TOP, M % BLOCKDIM_X, M, MPI_INT, &block_gap_x_t);
    MPI_Type_commit(&block_base_t);
    MPI_Type_commit(&block_gap_x_t);

    int* strip = (int*)malloc(sizeof(int) * BLOCKDIM_Y__LOCAL * M);

    if (world_rank == 0) {
        for (int i = 0; i < M; i += BLOCKDIM_X) {
            for (int j = 0; j < BLOCKDIM_Y__LOCAL; ++j) {
                for (int k = i; k < i + BLOCKDIM_X && k < M; ++k) {
                    strip[j * M + k] = j;
                }
            }
            if (world_size > 1) {
                MPI_Request req;
                MPI_Isend(strip + (BLOCKDIM_Y__LOCAL - HOLLOW_TOP) * M + i, 1, M - i < BLOCKDIM_X ? block_gap_x_t : block_base_t, 1, 42, MPI_COMM_WORLD, &req);
            }
        }
    }

    if (world_rank > 0) {
        MPI_Request* reqs = (MPI_Request*)malloc(sizeof(MPI_Request) * BLOCK_NUM);
        for (int i = 0; i < BLOCK_NUM; ++i) {
            MPI_Irecv(strip + i * BLOCKDIM_X, 1, M - i * BLOCKDIM_X < BLOCKDIM_X ? block_gap_x_t : block_base_t, world_rank - 1, 42, MPI_COMM_WORLD, reqs + i);
        }
        int done = 0;
        while (done < BLOCK_NUM) {
            for (int i = 0; i < BLOCK_NUM; ++i) {
                int flag = 0;
                if (reqs[i] != MPI_REQUEST_NULL) {
                    MPI_Test(reqs + i, &flag, MPI_STATUS_IGNORE);
                }
                if (flag) {
                    ++done;
                    for (int j = HOLLOW_TOP; j < BLOCKDIM_Y__LOCAL; ++j) {
                        for (int k = i * BLOCKDIM_X; k < (i + 1) * BLOCKDIM_X && k < M; ++k) {
                            strip[j * M + k] = strip[(j - 1) * M + k] + 1;
                        }
                    }
                    if (world_rank < world_size - 1) {
                        MPI_Request req;
                        MPI_Isend(strip + (BLOCKDIM_Y__LOCAL - HOLLOW_TOP) * M + i * BLOCKDIM_X, 1, M - i * BLOCKDIM_X < BLOCKDIM_X ? block_gap_x_t : block_base_t, world_rank + 1, 42, MPI_COMM_WORLD, &req);
                    }
                }
            }
        }
        free(reqs);
    }

    if (world_rank == world_size - 1) {
        int i = (world_rank == 0 ? 0 : HOLLOW_TOP);
        for (; i < BLOCKDIM_Y__LOCAL; ++i) {
            for (int j = 0; j < M; ++j) {
                fprintf(stderr, "%d ", strip[i * M + j]);
            }
            fprintf(stderr, "\n");
        }
    }

    // int* mx = (int*)malloc(sizeof(int) * N * M);

    // for (int i = 0; i < M; i += block_x) {
    //     if (world_rank > 0) {
    //         MPI_Recv(mx + (world_rank * block_y - 1) * M + i, 1, M - i < block_x ? block_gap_x_t : block_base_t, world_rank - 1, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     }
    //     for (int j = world_rank * block_y; j < ((world_rank + 1) * block_y < N ? (world_rank + 1) * block_y : N); ++j) {
    //         for (int k = i; k < (i + block_x < M ? i + block_x : M); ++k) {
    //             mx[j * M + k] = (world_rank == 0 ? j : mx[(j - 1) * M + k] + 1);
    //         }
    //     }
    //     if (world_rank < world_size - 1) {
    //         MPI_Send(mx + ((world_rank + 1) * block_y - 1) * M + i, 1, M - i < block_x ? block_gap_x_t : block_base_t, world_rank + 1, 42, MPI_COMM_WORLD);
    //     }
    // }

    // if (world_rank == world_size - 1) {
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < M; ++j) {
    //             fprintf(stderr, "%d ", mx[i * M + j]);
    //         }
    //         fprintf(stderr, "\n");
    //     }
    // }

    free(strip);

    MPI_Finalize();
}