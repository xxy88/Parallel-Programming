#include <mpi.h>
#include <omp.h>
#include "./headers/physics.h"
// #include "./headers/logger.h"
#include "./headers/shared.h"

int * send_counts;
int * displacement;
int my_rank;
int world_size;
int n_omp_threads;

void update_position() {
    #pragma omp parallel for
    for (int i = displacement[my_rank]; i < displacement[my_rank + 1]; ++i) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // check collision between two objects
        for (int j = 0; j < n_body; ++j) {
            if (i == j) continue;
            double distance_sqrt = get_distance(i, j, x, y);
            if (distance_sqrt * distance_sqrt <= 4 * radius2) {
                #pragma omp critical
                {
                    vx[i] = -vx[i];
                    vy[i] = -vy[i];
                    vx[j] = -vx[j];
                    vy[j] = -vy[j]; 
                }
            }
        }

        // check collision between object and wall
        if (x[i] <= 0 || x[i] >= bound_x) {
            #pragma omp critical
            {
                vx[i] = -vx[i];
            }
        }
        if (y[i] <= 0 || y[i] >= bound_y) {
            #pragma omp critical
            {
                vy[i] = -vy[i];
            }
        }
    }
}

void update_velocity() {
    #pragma omp parallel for
    for (int i = displacement[my_rank]; i < displacement[my_rank + 1]; i++) {
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
        #pragma omp critical
        {
            vx[i] += ax * dt;
            vy[i] += ay * dt;
        }
    }
}

void update_once() {
    omp_set_num_threads(n_omp_threads);
    update_velocity();
    for (int i = 0; i < world_size; ++i) {
        MPI_Bcast(&vx[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Bcast(&vy[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    update_position();
    for (int i = 0; i < world_size; ++i) {
        MPI_Bcast(&vx[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Bcast(&vy[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Bcast(&x[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Bcast(&y[displacement[i]], send_counts[i], MPI_DOUBLE, i, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void slave() {
    MPI_Bcast(m, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vx, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vy, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < n_iteration; i++) {
        update_once();
    }
}


void master() {
    generate_data(m, x, y, vx, vy, n_body);

    // Logger l = Logger("MPI", n_body, bound_x, bound_y);

    MPI_Bcast(m, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vx, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vy, n_body, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n_iteration; i++) {
        t1 = std::chrono::high_resolution_clock::now();

        update_once();

        t2 = std::chrono::high_resolution_clock::now();
        time_span = t2 - t1;

        printf("Iteration %d, elapsed time: %.3f\n", i, time_span);

        // l.save_frame(x, y);

#ifdef GUI
        glClear(GL_COLOR_BUFFER_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(2.0f);
        glBegin(GL_POINTS);
        double xi;
        double yi;
        for (int i = 0; i < n_body; i++) {
            xi = x[i];
            yi = y[i];
            glVertex2f(xi, yi);
        }
        glEnd();
        glFlush();
        glutSwapBuffers();
#else

#endif
    }
}


int main(int argc, char* argv[]) {
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_omp_threads = atoi(argv[3]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    m = new double[n_body];
    x = new double[n_body];
    y = new double[n_body];
    vx = new double[n_body];
    vy = new double[n_body];

    // partition data to each process
    int quotient = n_body / world_size;
    int remainder = n_body % world_size;
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
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
        glutInitWindowSize(800, 800);
        glutInitWindowPosition(0, 0);
        glutCreateWindow("N Body Simulation MPI-OpenMP Implementation");
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glMatrixMode(GL_PROJECTION);
        gluOrtho2D(0, bound_x, 0, bound_y);
#endif
        t3 = std::chrono::high_resolution_clock::now();
        master();
        t4 = std::chrono::high_resolution_clock::now();
        total_time = t4 - t3;
    } else {
        slave();
    }

    if (my_rank == 0) {
        printf("Total time: %f\n", total_time);
        printf("Student ID: 119020059\n"); // replace it with your student id
        printf("Name: Xinyu Xie\n"); // replace it with your name
        printf("Assignment 2: N Body Simulation MPI-OpenMP Implementation\n");
        printf("Number of Bodies: %d\n", n_body);
        printf("Number of Cores: %d\n", world_size);
    }

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;

    MPI_Finalize();

    return 0;
}

