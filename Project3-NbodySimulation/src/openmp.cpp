#include <omp.h>
#include "./headers/physics.h"
// #include "./headers/logger.h"
#include "./headers/shared.h"

int n_omp_threads;

void update_position(double* m, double* x, double* y, double* vx, double* vy, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // check collision between two objects
        for (int j = 0; j < n; ++j) {
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

void update_velocity(double* m, double* x, double* y, double* vx, double* vy, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double fx = 0;
        double fy = 0;
        // calculate force from all other objects
        for (int j = 0; j < n; ++j) {
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

void master() {
    double* m = new double[n_body];
    double* x = new double[n_body];
    double* y = new double[n_body];
    double* vx = new double[n_body];
    double* vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    // Logger l = Logger("OpenMP", n_body, bound_x, bound_y);

    for (int i = 0; i < n_iteration; i++){
        t1 = std::chrono::high_resolution_clock::now();

        //TODO: choose better threads configuration
        omp_set_num_threads(n_omp_threads);
        update_velocity(m, x, y, vx, vy, n_body);
        update_position(m, x, y, vx, vy, n_body);

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
        for (int i = 0; i < n_body; i++){
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

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
}


int main(int argc, char *argv[]){

    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);
    n_omp_threads = atoi(argv[3]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(800, 800);
    glutCreateWindow("N Body Simulation OpenMP Implementation");
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    gluOrtho2D(0, bound_x, 0, bound_y);
    #endif
    t3 = std::chrono::high_resolution_clock::now();
    master();
    t4 = std::chrono::high_resolution_clock::now();
    total_time = t4 - t3;

    printf("Total time: %f\n", total_time);
    printf("Student ID: 119020059\n"); // replace it with your student id
    printf("Name: Xinyu Xie\n"); // replace it with your name
    printf("Assignment 2: N Body Simulation OpenMP Implementation\n");
    printf("Number of Bodies: %d\n", n_body);
    printf("Number of Cores: %d\n", n_omp_threads);

    return 0;

}


