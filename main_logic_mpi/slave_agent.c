#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASTER 0
#define TAG_ORDER 1
#define TAG_MODEL 2
#define TAG_READY 3
#define TAG_FINISH 4
#define MAX_FILENAME 256

void execute_python_script(char *params) {
    char command[MAX_FILENAME + 50];
    sprintf(command, "python3 entrenar_agente.py %s", params);
    system(command);
}

void send_file_to_master(int master_rank, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *buffer = malloc(file_size);
    if (!buffer) {
        perror("Failed to allocate buffer");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fread(buffer, 1, file_size, file);
    fclose(file);

    MPI_Send(&file_size, 1, MPI_LONG, master_rank, TAG_MODEL, MPI_COMM_WORLD);
    MPI_Send(buffer, file_size, MPI_BYTE, master_rank, TAG_MODEL, MPI_COMM_WORLD);

    free(buffer);
}

int main(int argc, char *argv[]) {
    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

    char params[256];
    char model_filename[MAX_FILENAME];

    while (1) {
        MPI_Send(NULL, 0, MPI_BYTE, MASTER, TAG_READY, MPI_COMM_WORLD);

        MPI_Status status;
        MPI_Recv(params, 256, MPI_CHAR, MASTER, TAG_ORDER, MPI_COMM_WORLD, &status);
        if (strcmp(params, "FINISH") == 0) {
            break;
        }

        execute_python_script(params);

        sprintf(model_filename, "modelo_%s.tar", hostname);
        send_file_to_master(MASTER, model_filename);
    }

    MPI_Finalize();
    return 0;
}
