#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
// #include <time.h>

void fill_random(double* a, int n, int m) {
    // srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] = rand() / (double)RAND_MAX;
        }
    }
}

void dump_matrix(double* a, int n, int m, FILE* filp) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fprintf(filp, "%f ", a[i * m + j]);
        }
        fprintf(filp, "\n");
    }
}

int main(int argc, char** argv) 
{
    assert(argc == 3);

    // parse command line arguments
    MPI_Init(&argc, &argv);

    int n = atoi(argv[1]);// matrix dim.
    int p = atoi(argv[2]);// proc. num.

    assert(n % p == 0);
    int local_dim = n / p;

    // define row & col types to scatter data later
    MPI_Datatype row_type;
    MPI_Type_vector(local_dim, n, n, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    MPI_Datatype _col_type, col_type;
    MPI_Type_vector(n, local_dim, n, MPI_DOUBLE, &_col_type);
    MPI_Type_commit(&_col_type);
    MPI_Type_create_resized(_col_type, 0, local_dim * sizeof(double), &col_type);
    MPI_Type_commit(&col_type);

    FILE* log;

    // acquire world size and rank
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == p);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // initialize matrices a, b, c: c = a x b
    double* a = NULL;// matrix a
    double* b = NULL;// matrix b
    double* c = NULL;// matrix c = a x b
    if (world_rank == 0) {
        a = (double*) malloc(sizeof(double) * n * n);
        assert(a != NULL);
        b = (double*) malloc(sizeof(double) * n * n);
        assert(b != NULL);
        c = (double*) malloc(sizeof(double) * n * n);
        assert(c != NULL);

        fill_random(a, n, n);
        fill_random(b, n, n);

        // log = fopen("check", "w");
        // for (int i = 0; i < n; ++i)
        // {
        //     for (int j = 0; j < n; ++j)
        //     {
        //         c[i * n + j] = 0;
        //         for (int k = 0; k < n; ++k)
        //         {
        //             c[i * n + j] += a[i * n + k] * b[k * n + j];
        //         }
        //         fprintf(log, "%f ", c[i * n + j]);
        //     }
        //     fprintf(log, "\n");
        // }
        // fclose(log);
        
        log = fopen("a", "w");
        dump_matrix(a, n, n, log);// output 'a'
        fclose(log);

        log = fopen("b", "w");
        dump_matrix(b, n, n, log);// output 'b'
        fclose(log);
    }

    // scatter 'a' and 'b' to all processes
    double* local_a = (double*) malloc(sizeof(double) * local_dim * n);
    double* local_b = (double*) malloc(sizeof(double) * local_dim * n);
    double* local_c = (double*) malloc(sizeof(double) * local_dim * n);

    MPI_Scatter(a, 1, row_type, local_a, local_dim * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, 1, col_type, local_b, local_dim * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(a);
    free(b);

    // run calcs
    MPI_Request req[] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    double* swap_b = (double*) malloc(sizeof(double) * local_dim * n);// extra buffer to overlap comms with calcs
    double* _send_b;
    double* _recv_b;
    for (int step = 0; step < world_size; ++step) 
    {
        _recv_b = (step % 2 == 0 ? swap_b : local_b);
        _send_b = (step % 2 == 0 ? local_b : swap_b);
        
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        if (step < world_size - 1) {
            MPI_Irecv(_recv_b, 
                local_dim * n, 
                MPI_DOUBLE, 
                world_rank < world_size - 1 ? world_rank + 1 : 0, 
                42, MPI_COMM_WORLD, req);
            MPI_Isend(_send_b, 
                local_dim * n, 
                MPI_DOUBLE, 
                world_rank > 0 ? world_rank - 1 : world_size - 1, 
                42, MPI_COMM_WORLD, req + 1);
        }

        int displ = local_dim * ((world_rank + step) % world_size);
        for (int i = 0; i < local_dim; ++i) {
            for (int j = 0; j < local_dim; ++j) {
                local_c[displ + i * n + j] = 0;
            }

            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < local_dim; ++j) {
                    local_c[displ + i * n + j] += local_a[i * n + k] * _send_b[k * local_dim + j];
                }
            }
        }
    }

    free(local_a);
    free(local_b);
    free(swap_b);

    // accumulate result
    MPI_Gather(local_c, local_dim * n, MPI_DOUBLE, c, 1, row_type, 0, MPI_COMM_WORLD);

    free(local_c);

    // output 'c'
    if (world_rank == 0) {
        log = fopen("c", "w");
        dump_matrix(c, n, n, log);
        fclose(log);
    }

    free(c);

    MPI_Finalize();
}