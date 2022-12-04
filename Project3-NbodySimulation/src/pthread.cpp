#include <pthread.h>
#include "./headers/physics.h"
//#include "./headers/logger.h"
#include "./headers/shared.h"

int n_thd; // number of threads
pthread_mutex_t mutex; // mutex lock
pthread_barrier_t barrier;

void update_position(int start, int size) {
    for (int i = start; i < start + size; ++i) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // check collision between two objects
        for (int j = 0; j < n_body; ++j) {
            if (i == j) continue;
            double distance_sqrt = get_distance(i, j, x, y);
            if (distance_sqrt * distance_sqrt <= 4 * radius2) {
                vx[i] = -vx[i];
                vy[i] = -vy[i];
                pthread_mutex_lock(&mutex);
                vx[j] = -vx[j];
                vy[j] = -vy[j];
                pthread_mutex_unlock(&mutex);
            }
        }

        // check collision between object and wall
        if (x[i] <= 0 || x[i] >= bound_x) {
            vx[i] = -vx[i];
        }
        if (y[i] <= 0 || y[i] >= bound_y) {
            vy[i] = -vy[i];
        }
    }
}

void update_velocity(int start, int size) {
    for (int i = start; i < start + size; i++) {
        double fx = 0;
        double fy = 0;
        // calculate force from all other objects
        for (int j = 0; j < n_body; ++j) {
            if (i == j) continue;
            double distance_sqrt = get_distance(i, j, x, y);
            double f = gravity_const * m[i] * m[j] / (distance_sqrt * distance_sqrt + err);
            fx += f * (x[j] - x[i]) / distance_sqrt;
            fy += f * (y[j] - y[i]) / distance_sqrt;
        }
        // update velocity
        double ax = fx / m[i];
        double ay = fy / m[i];
        vx[i] += ax * dt;
        vy[i] += ay * dt;
    }
}


typedef struct {
    int start; // array start position of whole data array
    int size; // size of array
    int n_iterations;
} Args;


void* worker(void* args) {
    Args* my_arg = (Args*) args;
    int start = my_arg->start;
    int size = my_arg->size;
    int n_iterations = my_arg->n_iterations;

    for (int i = 0; i < n_iterations; i++)
    {
        update_velocity(start, size);
        update_position(start, size);
        pthread_barrier_wait(&barrier);
    }
    
    pthread_exit(NULL);
    return nullptr;
}


void master() {
    m = new double[n_body];
    x = new double[n_body];
    y = new double[n_body];
    vx = new double[n_body];
    vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

//    Logger l = Logger("Pthread", n_body, bound_x, bound_y);

    //TODO: assign jobs
    Args args[n_thd]; // arguments for all threads
    int size = n_body / n_thd;
    int remainder = n_body % n_thd;
    // allocate data array for each thread
    for (int thd = 0; thd < remainder; thd++) {
        args[thd].start = thd * (size + 1);
        args[thd].size = size + 1;
        args[thd].n_iterations = n_iteration;
    }
    for (int thd = remainder; thd < n_thd; thd++) {
        args[thd].start = thd * size + remainder;
        args[thd].size = size;
        args[thd].n_iterations = n_iteration;
    }

    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, n_thd + 1);

    pthread_t thds[n_thd]; // thread pool
    for (int thd = 0; thd < n_thd; thd++) pthread_create(&thds[thd], NULL, worker, &args[thd]);


    for (int i = 0; i < n_iteration; i++) {
        t1 = std::chrono::high_resolution_clock::now();
        
        pthread_barrier_wait(&barrier);

        t2 = std::chrono::high_resolution_clock::now();
        time_span = t2 - t1;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

//        l.save_frame(x, y);

#ifdef GUI
    glut_plot();
#else

#endif
    }

    for (int thd = 0; thd < n_thd; thd++) pthread_join(thds[thd], NULL);

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&mutex);
}


int main(int argc, char* argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_thd = atoi(argv[3]);

#ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(800, 800);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Pthread");
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0, bound_x, 0, bound_y);
#endif
    t3 = std::chrono::high_resolution_clock::now();
    master();
    t4 = std::chrono::high_resolution_clock::now();
    total_time = t4 - t3;

    printf("Total time: %f\n", total_time);
    printf("Student ID: 119020059\n"); // replace it with your student id
    printf("Name: Xinyu Xie\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation Pthreads Implementation\n");
    printf("Number of Bodies: %d\n", n_body);
    printf("Number of Cores: %d\n", n_thd);

    pthread_exit(NULL);

    return 0;
}

