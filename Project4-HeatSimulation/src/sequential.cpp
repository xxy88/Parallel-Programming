#include "physics.h"
#include "shared.h"


void update(float *data, float *new_data, bool* fire_area) {
    // update the temperature of each point, and store the result in `new_data` to avoid data racing
    for (int i = 1; i < (size - 1); i++){
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

void maintain_wall(float *data) {
    // maintain the temperature of the wall
    for (int i = 0; i < size; i++){
        data[i] = wall_temp;
        data[i * size] = wall_temp;
        data[i * size + size - 1] = wall_temp;
        data[(size - 1) * size + i] = wall_temp;
    }
}

void master(){
    data = new float[size * size];
    new_data = new float[size * size];
    fire_area = new bool[size * size];

    generate_fire_area(fire_area);
    initialize(data);

    while (count <= max_iteration) {
        t1 = std::chrono::high_resolution_clock::now();

        update(data, new_data, fire_area);
        maintain_wall(new_data);
        swap(data, new_data);

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

  delete[] data;
  delete[] new_data;
  delete[] fire_area;
}


int main(int argc, char* argv[]) {
    size = atoi(argv[1]);

    #ifdef GUI
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("Heat Distribution Simulation Sequential Implementation");
    gluOrtho2D(0, resolution, 0, resolution);
    #endif

    master();

    printf("Student ID: 119020059\n");
    printf("Name: Xinyu Xie\n");
    printf("Assignment 4: Heat Distribution Sequential Implementation\n");
    printf("Problem Size: %d\n", size);
    printf("Number of Cores: %d\n", 1);

    return 0;
}
