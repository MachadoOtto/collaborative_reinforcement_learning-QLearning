#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Tags
#define TAG_ORDER 1
#define TAG_MODEL_TO_SLAVE 2
#define TAG_MODEL_TO_MASTER 3
#define TAG_READY_SLAVE 4
#define TAG_FINISH 5

// Global constants
#define MAX_ORDERS 30
#define ENV_NAME "CartPole-v1"
#define TIMEOUT_TIME 300 // 5 minutes
#define N_GAMES 100
#define BASE_MODEL_PATH "./reinforcement_learner/outputs/models/"

// Master variables
#define MASTER 0

// Master logic
// Send an order to a slave
void send_order_to_slave(int slave_rank, int flag_no_model, FILE *model_file, long model_size) {
    // Check if there is a model to send
    // Send the order tag without the FINISH message
    char order_msg[16];
    sprintf(order_msg, "ORDER");
    MPI_Send(order_msg, strlen(order_msg) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
    if (flag_no_model) {
        // Send model size equal to 0
        long zero_model_size = 0;
        MPI_Send(&zero_model_size, 1, MPI_LONG, slave_rank, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD);
    } else {
        // Send model size and model
        MPI_Send(&model_size, 1, MPI_LONG, slave_rank, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD);
        MPI_Send(model_file, model_size, MPI_BYTE, slave_rank, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD);
    }
}

// Receive a model from a slave
FILE *receive_model_from_slave(int slave_rank, long slave_model_size) {
    FILE *slave_model_file = malloc(slave_model_size);
    if (!slave_model_file) {
        perror("Failed to allocate buffer for MPI_Recv");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Receive the model from the slave in chunks of 1MB
    int offset = 0;
    printf("MASTER Receiving model\n");
    while (offset < slave_model_size) {
        int chunk_size = slave_model_size - offset > 1048576 ? 1048576 : slave_model_size - offset;
        MPI_Recv(slave_model_file + offset, chunk_size, MPI_BYTE, slave_rank, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        offset += chunk_size;
    }
    return slave_model_file;
}

// Save a model to a file
void save_model_to_file(FILE *model_file, long model_size, char *model_filename) {
    FILE *file = fopen(model_filename, "wb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fwrite(model_file, 1, model_size, file);
    fclose(file);  
}

// Merge the models from the slaves using a python script
void merge_models(int slaveId) {
    char command[FILENAME_MAX + 50];
    sprintf(command, "python3 ./reinforcement_learner/master.py --env_name=%s --suffix='-model%d.pt,-modelogeneral.pt'", TIMEOUT_TIME, ENV_NAME, slaveId);
    system(command);
}

// Load a model from a file to a buffer, it returns the buffer and the size of the model
FILE *load_model(char *model_filename, long *model_size) {
    FILE *new_model_file = fopen(model_filename, "rb");
    if (!new_model_file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(new_model_file, 0, SEEK_END);
    long new_model_size = ftell(new_model_file);
    fseek(new_model_file, 0, SEEK_SET);
    FILE *model_file = malloc(new_model_size);
    if (!model_file) {
        perror("Failed to allocate buffer for MPI_Send");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fread(model_file, 1, new_model_size, new_model_file);
    fclose(new_model_file);
    *model_size = new_model_size;
    return model_file;
}

// Main master logic
void master() {
    MPI_Request request_slave_ready;
    MPI_Status status_slave_ready;
    MPI_Request request_slave_model;
    MPI_Status status_slave_model;
    int slave_rank;

    int flag_ready = 1;
    int flag_complete = 1;
    int flag_no_model = 1;
    int orders_complete = 0;

    FILE *model_file;
    long model_size;
    long slave_model_size;
    
    while (orders_complete < MAX_ORDERS) {
        // Check for slaves ready to work
        if (flag_ready) {
            MPI_Irecv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_READY_SLAVE, MPI_COMM_WORLD, &request_slave_ready);
            flag_ready = 0;
        } else {
            MPI_Test(&request_slave_ready, &flag_ready, &status_slave_ready);
            if (flag_ready) {
                // There is a slave ready
                slave_rank = status_slave_ready.MPI_SOURCE;
                if (orders_complete < MAX_ORDERS) {
                    printf("[MASTER] Sending order to slave %d\n", slave_rank);
                    send_order_to_slave(slave_rank, flag_no_model, model_file, model_size);
                } else {
                    char finish_msg[16];
                    sprintf(finish_msg, "FINISH");
                    MPI_Send(finish_msg, strlen(finish_msg) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
                }
            }
        }

        // Check for models from slaves
        if (flag_complete) {
            MPI_Irecv(&slave_model_size, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD, &request_slave_model);
            flag_complete = 0;
        } else {
            MPI_Test(&request_slave_model, &flag_complete, &status_slave_model);
            if (flag_complete) {
                // There is a model to receive
                printf("[MASTER] Receiving model from slave %d\n", status_slave_model.MPI_SOURCE);
                if (!flag_no_model) {
                    model_size = slave_model_size;
                    model_file = receive_model_from_slave(status_slave_model.MPI_SOURCE, slave_model_size);

                    // Save the model to a file (BASE_MODEL_PATH)
                    char model_filename[FILENAME_MAX];
                    sprintf(model_filename, "%s%s-modelogeneral.pt", BASE_MODEL_PATH, ENV_NAME);
                    save_model_to_file(model_file, model_size, model_filename);
                    orders_complete++;
                } else {
                    // If there is a model already, save it to a .pt file, and free it
                    FILE *slave_model = receive_model_from_slave(status_slave_model.MPI_SOURCE, slave_model_size);
                    char slave_model_filename[FILENAME_MAX];
                    sprintf(slave_model_filename, "%s%s-model%d.pt", BASE_MODEL_PATH, ENV_NAME, status_slave_model.MPI_SOURCE);
                    save_model_to_file(slave_model, slave_model_size, slave_model_filename);
                    free(slave_model);
                    
                    // Free the model file and save the merged model
                    merge_models(status_slave_model.MPI_SOURCE);
                    char new_model_filename[FILENAME_MAX];
                    sprintf(new_model_filename, "%s%s-model%d-modelogeneral_merged_model.pt", BASE_MODEL_PATH, ENV_NAME, status_slave_model.MPI_SOURCE);
                    free(model_file);
                    model_file = load_model(new_model_filename, &model_size);
                    printf("[MASTER] Merged model from slave %d\n", status_slave_model.MPI_SOURCE);
                    orders_complete++;
                }
                flag_no_model = 0;
                flag_complete = 1;
            }
        }
    }
}

// Slave logic
// Receive a model from the master and save it to a file
FILE *receive_model_from_master(long model_size, int rank) {
    // Receive the model
    FILE *model_file = malloc(model_size);
    if (!model_file) {
        perror("Failed to allocate buffer for MPI_Recv");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Recv(model_file, model_size, MPI_BYTE, MASTER, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Save the model to a file
    char model_filename[FILENAME_MAX];
    sprintf(model_filename, "%s%d-%s-model.pt", BASE_MODEL_PATH, rank, ENV_NAME);
    FILE *file = fopen(model_filename, "wb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fwrite(model_file, 1, model_size, file);
    fclose(file);
    free(model_file);
}

// Send a model to the master
void send_model_to_master(int master_rank, char *model_filename) {
    FILE *file = fopen(model_filename, "rb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = malloc(file_size);
    if (!buffer) {
        perror("Failed to allocate buffer for MPI_Send");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fread(buffer, 1, file_size, file);
    fclose(file);
    printf("Sending size\n");
    MPI_Send(&file_size, 1, MPI_LONG, master_rank, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD);
    // Send the model in chunks of 1MB
    int offset = 0;
    printf("Sending model\n");
    while (offset < file_size) {
        int chunk_size = file_size - offset > 1048576 ? 1048576 : file_size - offset;
        MPI_Send(buffer + offset, chunk_size, MPI_BYTE, master_rank, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD);
        offset += chunk_size;
    }
    free(buffer);
}

// Main slave logic
void slave(int rank) {
    char model_filename[FILENAME_MAX];
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    FILE *current_model;

    while (1) {
        // Notify the master that the slave is ready
        MPI_Send(NULL, 0, MPI_BYTE, MASTER, TAG_READY_SLAVE, MPI_COMM_WORLD);
        MPI_Get_processor_name(hostname, &hostname_len);
        printf("Slave %d (%s) is ready to work\n", rank, hostname);
        // Receive the order from the master
        char order_message[16];
        MPI_Recv(order_message, 16, MPI_CHAR, MASTER, TAG_ORDER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (strcmp(order_message, "FINISH") == 0) {
            // Finish the execution
            break;
        }
        // Receive size of model, if it is 0, there is no model to receive
        long model_size;
        MPI_Recv(&model_size, 1, MPI_LONG, MASTER, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char command[512];
        if (model_size > 0) {
            if (current_model) {
                free(current_model);
            }
            current_model = receive_model_from_master(model_size, rank);
            sprintf(command, "timeout %d python3 ./reinforcement_learner/slave.py --env_name=%s --n_games=%d --save_model=True --base_model=%s%d-%s-model.pt --file_name=%d-%s-model", TIMEOUT_TIME, ENV_NAME, N_GAMES, BASE_MODEL_PATH, rank, ENV_NAME, rank, ENV_NAME);
        } else {
            sprintf(command, "timeout %d python3 ./reinforcement_learner/slave.py --env_name=%s --n_games=%d --save_model=True --file_name=%d-%s-model", TIMEOUT_TIME, ENV_NAME, N_GAMES, rank, ENV_NAME);
        }
        // Execute the python script
        int status = system(command);
        if (status == 0) {
            // Script finalized before TIMEOUT_TIME
            // Send the model to the master
            sprintf(model_filename, "%s%d-%s-model.pt", BASE_MODEL_PATH, rank, ENV_NAME);
            printf("Slave %d (%s) finished the execution, sending the model to the master\n", rank, hostname);
            send_model_to_master(MASTER, model_filename);
        }
    }
}

// Main function
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        master();
    } else {
        slave(rank);
    }

    MPI_Finalize();
    return 0;
}
