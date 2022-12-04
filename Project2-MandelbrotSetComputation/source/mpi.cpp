#include "asg2.h"
#include <stdio.h>
#include <mpi.h>

void worker(Point* my_data, int my_size) {
    Point* p = my_data;
	for (int index = 0; index < my_size; index++){
		compute(p);
		p++;
	}
}

int main(int argc, char *argv[]) {
	if ( argc == 4 ) {
		X_RESN = atoi(argv[1]);
		Y_RESN = atoi(argv[2]);
		max_iteration = atoi(argv[3]);
	} else {
		X_RESN = 1000;
		Y_RESN = 1000;
		max_iteration = 100;
	}

	int rank;
	int world_size;
	int total_size = X_RESN * Y_RESN;
	/* computation part begin */
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	MPI_Datatype MPI_POINT;
	int blocklengths[] = {1, 1, 1};
	MPI_Aint displacements[] = {0, 4, 8};
	MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_FLOAT};
	MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);


	if (rank == 0) {
		#ifdef GUI
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
		glutInitWindowSize(500, 500); 
		glutInitWindowPosition(0, 0);
		glutCreateWindow("MPI");
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glMatrixMode(GL_PROJECTION);
		gluOrtho2D(0, X_RESN, 0, Y_RESN);
		glutDisplayFunc(plot);
		#endif
	}

	if (rank == 0) {
		t1 = std::chrono::high_resolution_clock::now();
		initData();
	}

	// partition data to each process
	int quotient = total_size / world_size;
	int remainder = total_size % world_size;
	int* send_counts = new int[world_size];
	int* displacement = new int[world_size];

	for (int i = 0; i < world_size; i++) {
        // number of elements allocated to each process
        send_counts[i] = quotient + (i < remainder ? 1 : 0);
    }

    displacement[0] = 0;
    for (int i = 1; i < world_size; i++) {
        // displacement of each process
        displacement[i] = displacement[i - 1] + send_counts[i - 1];
    }

    int my_size = send_counts[rank];
	Point* my_data = new Point[my_size]; // store elements of each process

    // distribute elements to each process
    MPI_Scatterv(data, send_counts, displacement, MPI_POINT, my_data, my_size, MPI_POINT, 0, MPI_COMM_WORLD);
    worker(my_data, my_size);
    // collect result from each process
    MPI_Gatherv(my_data, my_size, MPI_POINT, data, send_counts, displacement, MPI_POINT, 0, MPI_COMM_WORLD);

	delete[] send_counts;
	delete[] displacement;
	delete[] my_data;

	if (rank == 0) {
		t2 = std::chrono::high_resolution_clock::now();  
		time_span = t2 - t1;

		printf("Student ID: 119020059\n"); // replace it with your student id
		printf("Name: Xinyu Xie\n"); // replace it with your name
		printf("Assignment 2 MPI\n");
		printf("Run Time: %f seconds\n", time_span.count());
		printf("Problem Size: %d * %d, %d\n", X_RESN, Y_RESN, max_iteration);
		printf("Processing Speed: %f pixels/s\n", total_size / time_span.count());
		printf("Process Number: %d\n", world_size);
	}

	MPI_Finalize();
	/* computation part end */

	if (rank == 0){
		#ifdef GUI
		glutMainLoop();
		#endif
	}

	return 0;
}

