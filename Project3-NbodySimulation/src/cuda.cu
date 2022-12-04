#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "./headers/physics.h"
// #include "./headers/logger.h"
#include "./headers/shared.h"


int block_size = 512;


__global__ void update_position(double *x, double *y, double *vx, double *vy, int n) {
    int i_thd = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_thd < n) {
        int i = i_thd;
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;

        // check collision between two objects
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double distance_sqrt = sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]));
            if (distance_sqrt * distance_sqrt <= 4 * radius2) {
                vx[i] = -vx[i];
                vy[i] = -vy[i];
                vx[j] = -vx[j];
                vy[j] = -vy[j];
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

__global__ void update_velocity(double *m, double *x, double *y, double *vx, double *vy, int n) {
    int i_thd = blockDim.x * blockIdx.x + threadIdx.x;
    if (i_thd < n) {
        int i = i_thd;
        double fx = 0;
        double fy = 0;
        // calculate force from all other objects
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double distance_sqrt = sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]));
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

void master() {
    m = new double[n_body];
    x = new double[n_body];
    y = new double[n_body];
    vx = new double[n_body];
    vy = new double[n_body];

    generate_data(m, x, y, vx, vy, n_body);

    // Logger l = Logger("cuda", n_body, bound_x, bound_y);

    double *device_m;
    double *device_x;
    double *device_y;
    double *device_vx;
    double *device_vy;

    cudaMalloc(&device_m, n_body * sizeof(double));
    cudaMalloc(&device_x, n_body * sizeof(double));
    cudaMalloc(&device_y, n_body * sizeof(double));
    cudaMalloc(&device_vx, n_body * sizeof(double));
    cudaMalloc(&device_vy, n_body * sizeof(double));

    cudaMemcpy(device_m, m, n_body * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, x, n_body * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, n_body * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vx, vx, n_body * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vy, vy, n_body * sizeof(double), cudaMemcpyHostToDevice);

    int n_block = n_body / block_size + 1;

    for (int i = 0; i < n_iteration; i++){
        t1 = std::chrono::high_resolution_clock::now();

        update_velocity<<<n_block, block_size>>>(device_m, device_x, device_y, device_vx, device_vy, n_body);
        update_position<<<n_block, block_size>>>(device_x, device_y, device_vx, device_vy, n_body);

        cudaMemcpy(x, device_x, n_body * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(y, device_y, n_body * sizeof(double), cudaMemcpyDeviceToHost);

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

    cudaFree(device_m);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_vx);
    cudaFree(device_vy);

    delete[] m;
    delete[] x;
    delete[] y;
    delete[] vx;
    delete[] vy;
    
}


int main(int argc, char *argv[]){
    
    n_body = atoi(argv[1]);
    n_iteration = atoi(argv[2]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(800, 800);
    glutCreateWindow("N Body Simulation CUDA Implementation");
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
    printf("Assignment 2: N Body Simulation CUDA Implementation\n");
    printf("Number of Bodies: %d\n", n_body);
    printf("Number of Cores: %d\n", n_body);

    return 0;

}


