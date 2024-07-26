#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASTER 0
#define TAG_ORDER 1
#define TAG_MODEL 2
#define TAG_READY 3
#define TAG_FINISH 4
#define MAX_ORDERS 30

void generate_initial_model() {
    system("python3 generar_modelo_inicial.py");
}

void receive_model_from_slave(int slave_rank, const char *hostname) {
    MPI_Status status;
    long file_size;

    MPI_Recv(&file_size, 1, MPI_LONG, slave_rank, TAG_MODEL, MPI_COMM_WORLD, &status);

    char *buffer = malloc(file_size);
    if (!buffer) {
        perror("Failed to allocate buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Recv(buffer, file_size, MPI_BYTE, slave_rank, TAG_MODEL, MPI_COMM_WORLD, &status);

    char model_filename[MAX_FILENAME];
    sprintf(model_filename, "modelo_%s.tar", hostname);
    FILE *file = fopen(model_filename, "wb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fwrite(buffer, 1, file_size, file);
    fclose(file);

    free(buffer);

    char command[MAX_FILENAME + 50];
    sprintf(command, "python3 unificar_modelos.py modelogeneral.tar %s", model_filename);
    system(command);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int orders_sent = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        generate_initial_model();

        while (orders_sent < MAX_ORDERS) {
            MPI_Status status;
            int slave_rank;
            char hostname[MPI_MAX_PROCESSOR_NAME];

            MPI_Recv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_READY, MPI_COMM_WORLD, &status);
            slave_rank = status.MPI_SOURCE;

            MPI_Get_processor_name(hostname, &status.count);

            if (orders_sent < MAX_ORDERS) {
                char params[256];
                sprintf(params, "epsilon=0.1 gamma=0.99 lr=0.001");
                MPI_Send(params, strlen(params) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
                orders_sent++;
            } else {
                char finish_msg[] = "FINISH";
                MPI_Send(finish_msg, strlen(finish_msg) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
            }

            receive_model_from_slave(slave_rank, hostname);
        }
    }

    MPI_Finalize();
    return 0;
}
