#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cstdlib>

#include "Windows.h"
#endif

#ifdef GUI
#if defined(__APPLE__)

#include <GLUT/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>

#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

int n_body;
int n_iteration;
static double* m;
static double* x;
static double* y;
static double* vx;
static double* vy;

/* to keep track of time */
std::chrono::high_resolution_clock::time_point t1;
std::chrono::high_resolution_clock::time_point t2;
std::chrono::duration<double> time_span;
std::chrono::high_resolution_clock::time_point t3;
std::chrono::high_resolution_clock::time_point t4;
std::chrono::duration<double> total_time;


void generate_data(double* m, double* x, double* y, double* vx, double* vy, int n) {
    // TODO: Generate proper initial position and mass for better visualization
    srand((unsigned) time(NULL));
    for (int i = 0; i < n; i++) {
        m[i] = rand() % max_mass + 1.0f;
        x[i] = 2000.0f + rand() % (bound_x / 4);
        y[i] = 2000.0f + rand() % (bound_y / 4);
        vx[i] = 0.0f;
        vy[i] = 0.0f;
    }
}

double get_distance(int i, int j, double* x, double* y) {
    return sqrt((x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]));
}

#ifdef GUI
void glut_plot(){
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
}
#endif
