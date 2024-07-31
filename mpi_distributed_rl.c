// MPI Distributed Reinforcement Learning for Gym environments
// Description: This file contains the main logic for the distributed reinforcement learning using MPI.
// Authors:
//   - Guido Dinello ( 5.031.022-5 )
//   - Jorge Machado ( 4.876.616-9 )
// HPC - High Performance Computing - 2024
// Facultad de Ingeniería - Universidad de la República, Uruguay

// Libraries
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
#define TAG_SIZE_MODEL_TO_SLAVE 6
#define TAG_SIZE_MODEL_TO_MASTER 7

// Slaves Statuses
#define SLAVE_READY 1
#define SLAVE_WORKING 2
#define SLAVE_FINISHED 3
#define SLAVE_IDLE 4

// Master variables
#define MASTER 0

// Global constants (MODIFY HERE)
#define MAX_ORDERS 30
#define ENV_NAME "CartPole-v1"
#define TIMEOUT_TIME 300 // 5 minutes
#define N_GAMES 100
#define BASE_MODEL_PATH "./reinforcement_learner/outputs/models/"

// MASTER LOGIC //
// Send an order to a slave
void send_order_to_slave(int slave_rank, int flag_no_model) {
    // Check if there is a model to send
    // Send the order tag without the FINISH message
    char order_msg[16];
    sprintf(order_msg, "ORDER");
    MPI_Send(order_msg, strlen(order_msg) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
    if (flag_no_model) {
        // Send model size equal to 0
        long zero_model_size = 0;
        MPI_Send(&zero_model_size, 1, MPI_LONG, slave_rank, TAG_SIZE_MODEL_TO_SLAVE, MPI_COMM_WORLD);
    } else {
        // Send model size and model
        char model_filename[FILENAME_MAX];
        sprintf(model_filename, "%s%s-master_model.pt", BASE_MODEL_PATH, ENV_NAME);
        printf("[MASTER] Sending model %s to slave %d\n", model_filename, slave_rank);
        FILE *model_file = fopen(model_filename, "rb");
        if (!model_file) {
            perror("Failed to open file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fseek(model_file, 0, SEEK_END);
        long model_size = ftell(model_file);
        fseek(model_file, 0, SEEK_SET);
        MPI_Send(&model_size, 1, MPI_LONG, slave_rank, TAG_SIZE_MODEL_TO_SLAVE, MPI_COMM_WORLD);
        char *model_buffer = malloc(1000);
        if (!model_buffer) {
            perror("Failed to allocate buffer for MPI_Send");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int i = 0;
        while (i < model_size) {
            int chunk_size = 1000;
            if (i + chunk_size > model_size) {
                chunk_size = model_size - i;
            }
            fread(model_buffer, 1, chunk_size, model_file);
            MPI_Send(model_buffer, chunk_size, MPI_CHAR, slave_rank, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD);
            i += chunk_size;
        }
        free(model_buffer);
        fclose(model_file);
        printf("[MASTER] Sent model to slave %d\n", slave_rank);
    }
}

// Receive a model from a slave and save it to a file
void receive_model_from_slave(int slave_rank, long slave_model_size, int not_exist_model) {
    char *slave_model_file = malloc(1000);
    if (!slave_model_file) {
        perror("Failed to allocate buffer for MPI_Recv");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Open the file to save the model
    char slave_model_filename[FILENAME_MAX];
    if (not_exist_model) {
        sprintf(slave_model_filename, "%s%s-master_model.pt", BASE_MODEL_PATH, ENV_NAME);
    } else {
        sprintf(slave_model_filename, "%s%s-slave_model.pt", BASE_MODEL_PATH, ENV_NAME);
    }
    printf("[MASTER] Generating file slave_model.pt for slave %d into %s\n", slave_rank, slave_model_filename);
    FILE *file = fopen(slave_model_filename, "wb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Receive the model from the slave and write appending to the file
    int i = 0;
    while (i < slave_model_size) {
        int chunk_size = 1000;
        if (i + chunk_size > slave_model_size) {
            chunk_size = slave_model_size - i;
        }
        MPI_Recv(slave_model_file, chunk_size, MPI_CHAR, slave_rank, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fwrite(slave_model_file, 1, chunk_size, file);
        i += chunk_size;
    }
    free(slave_model_file);
    fclose(file);
}

// Merge the models from the slaves using a python script
void merge_models(int slaveId) {
    char command[FILENAME_MAX + 50];
    sprintf(command, "python3 ./reinforcement_learner/master.py --env_name=%s --suffix='-slave_model,-master_model'", ENV_NAME);
    printf("[MASTER] %s\n", command);
    system(command);
    char master_model_filename[FILENAME_MAX];
    sprintf(master_model_filename, "%s%s-master_model.pt", BASE_MODEL_PATH, ENV_NAME);
    char merged_model_filename[FILENAME_MAX];
    sprintf(merged_model_filename, "%s%s-slave_model-master_model_merged_model.pt", BASE_MODEL_PATH, ENV_NAME);
    // Save the new merged model to the master-model file
    FILE *file_master = fopen(master_model_filename, "wb");
    if (!file_master) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    FILE *file_merged = fopen(merged_model_filename, "rb");
    if (!file_merged) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(file_merged, 0, SEEK_END);
    long merged_model_size = ftell(file_merged);
    fseek(file_merged, 0, SEEK_SET);
    char *merged_model_buffer = malloc(merged_model_size);
    if (!merged_model_buffer) {
        perror("Failed to allocate buffer for MPI_Send");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fread(merged_model_buffer, 1, merged_model_size, file_merged);
    fwrite(merged_model_buffer, 1, merged_model_size, file_master);
    free(merged_model_buffer);
    fclose(file_master);
    fclose(file_merged);
}

// Main master logic
void master(int world_size) {
    MPI_Request request_slave_ready;
    MPI_Status status_slave_ready;
    MPI_Request request_slave_model;
    MPI_Status status_slave_model;
    int slave_rank;
    int slave_model_size;

    int flag_ready = 1;
    int flag_ready_slave = 0;
    int flag_complete = 1;
    int flag_complete_slave = 0;
    int flag_no_model = 1;
    int orders_complete = 0;

    // Array with slaves status
    int slaves_status[world_size];
    for (int i = 1; i < world_size; i++) {
        slaves_status[i] = SLAVE_IDLE;
    }
    
    while (orders_complete < MAX_ORDERS) {
        // Check for slaves ready to work
        if (flag_ready) {
            MPI_Irecv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_READY_SLAVE, MPI_COMM_WORLD, &request_slave_ready);
            flag_ready = 0;
        } else {
            MPI_Test(&request_slave_ready, &flag_ready_slave, &status_slave_ready);
            if (flag_ready_slave) {
                slaves_status[status_slave_ready.MPI_SOURCE] = SLAVE_READY;
                flag_ready_slave = 0;
                // There is a slave ready
                slave_rank = status_slave_ready.MPI_SOURCE;
                if (orders_complete < MAX_ORDERS) {
                    printf("[MASTER] Sending order to slave %d\n", slave_rank);
                    send_order_to_slave(slave_rank, flag_no_model);
                    slaves_status[slave_rank] = SLAVE_WORKING;
                } else {
                    char finish_msg[16];
                    sprintf(finish_msg, "FINISH");
                    MPI_Send(finish_msg, strlen(finish_msg) + 1, MPI_CHAR, slave_rank, TAG_ORDER, MPI_COMM_WORLD);
                    slaves_status[slave_rank] = SLAVE_FINISHED;
                }
                flag_ready = 1;
            }
        }

        // Check for models from slaves
        if (flag_complete) {
            MPI_Irecv(&slave_model_size, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_SIZE_MODEL_TO_MASTER, MPI_COMM_WORLD, &request_slave_model);
            flag_complete = 0;
        } else {
            MPI_Test(&request_slave_model, &flag_complete_slave, &status_slave_model);
            if (flag_complete_slave) {
                slaves_status[status_slave_model.MPI_SOURCE] = SLAVE_IDLE;
                flag_complete_slave = 0;
                // There is a model to receive
                printf("[MASTER] Receiving model from slave %d\n", status_slave_model.MPI_SOURCE);
                if (flag_no_model) {
                    receive_model_from_slave(status_slave_model.MPI_SOURCE, slave_model_size, 1);
                    flag_no_model = 0;
                } else {
                    // If there is a model already, save it to a .pt file, and free it
                    receive_model_from_slave(status_slave_model.MPI_SOURCE, slave_model_size, 0);
                    // Free the model file and save the merged model
                    printf("[MASTER] Merging model from slave %d\n", status_slave_model.MPI_SOURCE);
                    merge_models(status_slave_model.MPI_SOURCE);
                }
                orders_complete++;
                printf("[MASTER] Complete receiving model from slave %d\n", status_slave_model.MPI_SOURCE);
                flag_complete = 1;
                printf("[MASTER] Order %d complete! %d/%d\n", orders_complete, orders_complete, MAX_ORDERS);
            }
        }
    }

    // Send finish message to all slaves that are not finished
    for (int i = 1; i < world_size; i++) {
        if (slaves_status[i] != SLAVE_FINISHED) {
            char finish_msg[16];
            sprintf(finish_msg, "FINISH");
            MPI_Send(finish_msg, strlen(finish_msg) + 1, MPI_CHAR, i, TAG_ORDER, MPI_COMM_WORLD);
        }
    }

    printf("[MASTER] Finished all orders\n");
    printf("[MASTER] Waiting for slaves to finish\n");
    // Wait for all slaves to finish with MPI_Barrier
    MPI_Barrier(MPI_COMM_WORLD);
    printf("[MASTER] All slaves finished\n");
}

// SLAVE LOGIC //
// Receive a model from the master and save it to a file
void receive_model_from_master(long model_size, int rank) {
    // Receive the model
    char *buffer = malloc(1000);
    if (!buffer) {
        perror("Failed to allocate buffer for MPI_Recv");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    char model_filename[FILENAME_MAX];
    sprintf(model_filename, "%s%d-%s-model.pt", BASE_MODEL_PATH, rank, ENV_NAME);
    printf("[SLAVE %d] Saving file model into %s\n", rank, model_filename);
    FILE *file = fopen(model_filename, "wb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int i = 0;
    while (i < model_size) {
        int chunk_size = 1000;
        if (i + chunk_size > model_size) {
            chunk_size = model_size - i;
        }
        MPI_Recv(buffer, chunk_size, MPI_CHAR, MASTER, TAG_MODEL_TO_SLAVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fwrite(buffer, 1, chunk_size, file);
        i += chunk_size;
    }
    free(buffer);
    fclose(file);
    printf("[SLAVE %d] Received model from master\n", rank);
}

// Send a model to the master
void send_model_to_master(int rank) {
    char *buffer = malloc(1000);
    if (!buffer) {
        perror("Failed to allocate buffer for MPI_Send");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    char model_filename[FILENAME_MAX];
    sprintf(model_filename, "%s%d-%s-model.pt", BASE_MODEL_PATH, rank, ENV_NAME);
    printf("[SLAVE %d] Sending model %s to master\n", rank, model_filename);
    FILE *file = fopen(model_filename, "rb");
    if (!file) {
        perror("Failed to open file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    MPI_Send(&file_size, 1, MPI_LONG, MASTER, TAG_SIZE_MODEL_TO_MASTER, MPI_COMM_WORLD);
    int i = 0;
    while (i < file_size) {
        int chunk_size = 1000;
        if (i + chunk_size > file_size) {
            chunk_size = file_size - i;
        }
        fread(buffer, 1, chunk_size, file);
        MPI_Send(buffer, chunk_size, MPI_CHAR, MASTER, TAG_MODEL_TO_MASTER, MPI_COMM_WORLD);
        i += chunk_size;
    }
    free(buffer);
    fclose(file);
    printf("[SLAVE %d] Sent model to master\n", rank);
}

// Main slave logic
void slave(int rank) {
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    double idle_time = 0;
    int count_completed_orders = 0;

    while (1) {
        // Notify the master that the slave is ready
        MPI_Send(NULL, 0, MPI_BYTE, MASTER, TAG_READY_SLAVE, MPI_COMM_WORLD);
        MPI_Get_processor_name(hostname, &hostname_len);
        printf("[SLAVE %d] Waiting for orders\n", rank);
        // Receive the order from the master
        char order_message[16];
        // Record idle time
        double start = MPI_Wtime();        
        MPI_Recv(order_message, 16, MPI_CHAR, MASTER, TAG_ORDER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double end = MPI_Wtime();
        idle_time += end - start;
        if (strcmp(order_message, "FINISH") == 0) {
            // Finish the execution
            printf("[SLAVE %d] Finished the execution with idle time %f, and completed %d orders\n", rank, idle_time, count_completed_orders);
            break;
        }
        // Receive size of model, if it is 0, there is no model to receive
        long model_size;
        MPI_Recv(&model_size, 1, MPI_LONG, MASTER, TAG_SIZE_MODEL_TO_SLAVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        char command[512];
        if (model_size > 0) {
            printf("[SLAVE %d] Receiving model from master\n", rank);
            receive_model_from_master(model_size, rank);
            sprintf(command, "timeout %d python3 ./reinforcement_learner/slave.py --env_name=%s --n_games=%d --save_model=True --base_model=%s%d-%s-model.pt --file_name=%d-%s-model", TIMEOUT_TIME, ENV_NAME, N_GAMES, BASE_MODEL_PATH, rank, ENV_NAME, rank, ENV_NAME);
        } else {
            sprintf(command, "timeout %d python3 ./reinforcement_learner/slave.py --env_name=%s --n_games=%d --save_model=True --file_name=%d-%s-model", TIMEOUT_TIME, ENV_NAME, N_GAMES, rank, ENV_NAME);
        }
        // Execute the python script
        printf("[SLAVE %d] Executing reinforcement learning\n", rank);
        int status = system(command);
        if (status == 0) {
            // Script finalized before TIMEOUT_TIME
            // Send the model to the master
            printf("[SLAVE %d] Finished reinforcement learning. Sending model to master\n", rank); 
            send_model_to_master(rank);
            count_completed_orders++;
        }
    }
    // Wait for all slaves to finish with MPI_Barrier
    MPI_Barrier(MPI_COMM_WORLD);
}

// MAIN //
int main(int argc, char *argv[]) {
    printf("MPI Distributed Reinforcement Learning\n");
    printf("Authors:\n");
    printf("  - Guido Dinello ( 5.031.022-5 )\n");
    printf("  - Jorge Machado ( 4.876.616-9 )\n");
    printf("HPC - High Performance Computing - 2024\n");
    printf("Facultad de Ingeniería - Universidad de la República, Uruguay\n");
    printf("\n");

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_games = N_GAMES * MAX_ORDERS;
    printf(" - Environment: %s\n", ENV_NAME);
    printf(" - Number of games: %d\n", num_games);
    printf(" - Timeout time: %d seconds\n", TIMEOUT_TIME);
    
    if (rank == MASTER) {
        master(size);
    } else {
        slave(rank);
    }

    MPI_Finalize();
    return 0;
}
