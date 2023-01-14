#include "physics.h"
#include "shared.h"
#include <pthread.h>

int n_thd; // number of threads
pthread_barrier_t main_barrier;
pthread_barrier_t worker_barrier;

void update(float *data, float *new_data, int start, int local_size) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    for (int i = start; i < start + local_size; i++){
        for (int j = 1; j < (size - 1); j++){
            int idx = i * size + j;
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

void maintain_wall(float *data, int start, int local_size) {
    // maintain the temperature of the wall
    for (int i = start; i < start + local_size; i++){
        data[i] = wall_temp;
        data[i * size] = wall_temp;
        data[i * size + size - 1] = wall_temp;
        data[(size - 1) * size + i] = wall_temp;
    }
}


typedef struct {
    int start[3]; // array start position, 3 cutting methods
    int size[3]; // array size, 3 cutting methods
} Args;


void* worker(void* args) {
    Args* my_arg = (Args*) args;
    int* start = my_arg->start;
    int* size = my_arg->size;

    while (count <= max_iteration) {
        update(data, new_data, start[2], size[2]);
        pthread_barrier_wait(&worker_barrier);
        maintain_wall(new_data, start[1], size[1]);
        pthread_barrier_wait(&main_barrier);
        pthread_barrier_wait(&main_barrier);
    }

    pthread_exit(NULL);
    return nullptr;
}


void master() {
    data = new float[size * size];
    new_data = new float[size * size];
    fire_area = new bool[size * size];

    generate_fire_area(fire_area);
    initialize(data);

    // assign jobs
    Args *args = new Args[n_thd]; // arguments for all threads

    // split data 0 to size * size
    int local_size0 = size * size / n_thd;
    int remainder0 = size * size % n_thd;
    for (int thd = 0; thd < remainder0; thd++) {
        args[thd].start[0] = thd * (local_size0 + 1);
        args[thd].size[0] = local_size0 + 1;
    }
    for (int thd = remainder0; thd < n_thd; thd++) {
        args[thd].start[0] = thd * local_size0 + remainder0;
        args[thd].size[0] = local_size0;
    }

    // split data 0 to size
    int local_size1 = size / n_thd;
    int remainder1 = size % n_thd;
    for (int thd = 0; thd < remainder1; thd++) {
        args[thd].start[1] = thd * (local_size1 + 1);
        args[thd].size[1] = local_size1 + 1;
    }
    for (int thd = remainder1; thd < n_thd; thd++) {
        args[thd].start[1] = thd * local_size1 + remainder1;
        args[thd].size[1] = local_size1;
    }

    // split data 1 to size - 1
    int local_size2 = (size - 2) / n_thd;
    int remainder2 = (size - 2) % n_thd;
    for (int thd = 0; thd < remainder2; thd++) {
        args[thd].start[2] = 1 + thd * (local_size2 + 1);
        args[thd].size[2] = local_size2 + 1;
    }
    for (int thd = remainder2; thd < n_thd; thd++) {
        args[thd].start[2] = 1 + thd * local_size2 + remainder2;
        args[thd].size[2] = local_size2;
    }

    pthread_barrier_init(&main_barrier, NULL, n_thd + 1);
    pthread_barrier_init(&worker_barrier, NULL, n_thd);

    pthread_t* thds = new pthread_t[n_thd]; // thread pool
    for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);


    while (count <= max_iteration) {
        t1 = std::chrono::high_resolution_clock::now();

        pthread_barrier_wait(&main_barrier); // wait for update once
        swap(data, new_data);
        pthread_barrier_wait(&main_barrier); // wait for deep copy

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

    for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);

    printf("Converge after %d iterations, elapsed time: %.6f, average computation time: %.6f\n", count-1, total_time, (double) total_time / (count-1));

    delete[] data;
    delete[] new_data;
    delete[] fire_area;
    delete[] args;
    delete[] thds;

    pthread_barrier_destroy(&main_barrier);
    pthread_barrier_destroy(&worker_barrier);
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);
    n_thd = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("Heat Distribution Simulation Pthreads Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 119020059\n");
    printf("Name: Xinyu Xie\n");
    printf("Assignment 4: Heat Distribution Pthreads Implementation\n");
    printf("Problem Size: %d\n", size);
    printf("Number of Cores: %d\n", n_thd);

    return 0;
}

