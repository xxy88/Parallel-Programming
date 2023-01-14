#include <mpi.h>
#include "physics.h"
#include "shared.h"

int * send_counts;
int * displacement;
int my_rank;
int world_size;


void update(float *data, float *new_data) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    for (int i = displacement[my_rank]; i < displacement[my_rank + 1]; i++){
        for (int j = 1; j < (size - 1); j++){
            int idx = (i + 1) * size + j;
            if (fire_area[idx]) new_data[idx] = fire_temp;
            else {
                float up = data[idx - size];
                float down = data[idx + size];
                float left = data[idx - 1];
                float right = data[idx + 1];
                float new_val = (up + down + left + right) / 4;
                new_data[idx] = new_val;
            }
        }
    }
}

void maintain_wall(float *data) {
    // maintain the temperature of the wall
    for (int i = 0; i < size; i++){
        data[i] = wall_temp;
        data[i * size] = wall_temp;
        data[i * size + size - 1] = wall_temp;
        data[(size - 1) * size + i] = wall_temp;
    }
}


void update_once() {
    update(data, new_data);

    // send the last row to next rank
    if (my_rank != world_size - 1) {
        float* last_row = &new_data[displacement[my_rank + 1] * size] - size;
        MPI_Send(last_row, size, MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&new_data[displacement[my_rank + 1] * size], size, MPI_FLOAT, my_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // send the previous row the previous rank
    if (my_rank != 0) {
        float* first_neighbor_row = &new_data[displacement[my_rank] * size] - size;
        MPI_Recv(first_neighbor_row, size, MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&new_data[displacement[my_rank] * size], size, MPI_FLOAT, my_rank - 1, 0, MPI_COMM_WORLD);
    }

    maintain_wall(new_data);
    swap(data, new_data);
}

void slave() {
    while (count <= max_iteration) {
        update_once();
        count++;
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void master() {
    while (count <= max_iteration) {
        t1 = std::chrono::high_resolution_clock::now();

        update_once();
        MPI_Barrier(MPI_COMM_WORLD);

        t2 = std::chrono::high_resolution_clock::now();
        double this_time = std::chrono::duration<double>(t2 - t1).count();
        if (DEBUG) printf("Iteration %d, elapsed time: %.6f\n", count, this_time);
        total_time += this_time;

#ifdef GUI
        data2pixels(data, pixels);
        plot(pixels);
#endif

        count++;
    }
    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    data = new float[size * size];
    new_data = new float[size * size];
    fire_area = new bool[size * size];

    generate_fire_area(fire_area);
    initialize(data);

    // partition data to each process
    int quotient = (size - 2) / world_size;
    int remainder = (size - 2) % world_size;
    send_counts = new int[world_size];
    displacement = new int[world_size + 1];
    for (int i = 0; i < world_size; i++) {
        // number of elements allocated to each process
        send_counts[i] = quotient + (i < remainder ? 1 : 0);
    }
    displacement[0] = 0;
    for (int i = 1; i <= world_size; i++) {
        // displacement of each process
        displacement[i] = displacement[i - 1] + send_counts[i - 1];
    }


    if (my_rank == 0) {
        #ifdef GUI
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(window_size, window_size);
        glutCreateWindow("Heat Distribution Simulation MPI Implementation");
        gluOrtho2D(0, resolution, 0, resolution);
        #endif
        master();
    } else {
        slave();
    }

    if (my_rank == 0) {
        printf("Student ID: 119020059\n");
        printf("Name: Xinyu Xie\n");
        printf("Assignment 4: Heat Distribution MPI Implementation\n");
        printf("Problem Size: %d\n", size);
        printf("Number of Cores: %d\n", world_size);
    }

    delete[] data;
    delete[] new_data;
    delete[] fire_area;

    MPI_Finalize();

    return 0;
}

