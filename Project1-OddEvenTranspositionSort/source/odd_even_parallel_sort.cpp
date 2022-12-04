#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mpi.h>

int odd_even_sort(int* my_elements_arr, int my_size, int rank, int total_num_elements, int last_p, MPI_Comm comm) {
    int send_temp = 0;
    int recv_temp = 0;
    // the process rank of my left and right
    int my_right_proc = (rank + 1) % last_p;
    int my_left_proc = (rank + last_p - 1) % last_p;

    // fixed total sorting steps
    for (int i = 0; i <= total_num_elements; i++) {
        if (i % 2 == 0) {
            // odd sort
            for (int j = 0; j < my_size - 1; j += 2) {
                if (my_elements_arr[j] > my_elements_arr[j + 1]) {
                    std::swap(my_elements_arr[j], my_elements_arr[j + 1]);
                }
            }
        }
        else {
            // even sort
            for (int j = 1; j < my_size - 1; j += 2) {
                if (my_elements_arr[j] > my_elements_arr[j + 1]) {
                    std::swap(my_elements_arr[j], my_elements_arr[j + 1]);
                }
            }
            if ((rank != 0) && (rank <= last_p - 1)) {
                // if not the first process, send the first element to the left
                send_temp = my_elements_arr[0];
                MPI_Send(&send_temp, 1, MPI_INT, my_left_proc, 0, comm);
                MPI_Recv(&recv_temp, 1, MPI_INT, my_left_proc, 0, comm, MPI_STATUS_IGNORE);
                if (recv_temp > my_elements_arr[0]) {
                    my_elements_arr[0] = recv_temp;
                }
            }
            if ((rank != last_p - 1) && (rank <= last_p - 1)) {
                // if not the last process, send the last element to the right
                send_temp = my_elements_arr[my_size - 1];
                MPI_Recv(&recv_temp, 1, MPI_INT, my_right_proc, 0, comm, MPI_STATUS_IGNORE);
                MPI_Send(&send_temp, 1, MPI_INT, my_right_proc, 0, comm);
                if (recv_temp < my_elements_arr[my_size - 1]) {
                    my_elements_arr[my_size - 1] = recv_temp;
                }
            }
        }
    }

    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int num_elements;  // number of elements to be sorted

    num_elements = atoi(argv[1]);  // convert command line argument to num_elements

    int elements[num_elements];  // store elements
    int sorted_elements[num_elements];  // store sorted elements

    if (rank == 0) {  // read inputs from file (master process)
        std::ifstream input(argv[2]);
        int element;
        int i = 0;
        while (input >> element) {
            elements[i] = element;
            i++;
        }
        std::cout << "actual number of elements:" << i << std::endl;
    }

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    if (rank == 0) {
        t1 = std::chrono::high_resolution_clock::now();  // record time
    }

    /* TODO BEGIN
        Implement parallel odd even transposition sort
        Code in this block is not a necessary.
        Replace it with your own code.
        Useful MPI documentation: https://rookiehpc.github.io/mpi/docs
    */

    int quotient = num_elements / world_size;
    int remainder = num_elements % world_size;
    int* send_counts = new int[world_size];
    int* displacement = new int[world_size];
    // last p is the number of processes, or number of elements when # elements < # processes
    int last_p = (world_size < num_elements) ? world_size : num_elements;

    for (int i = 0; i < world_size; i++) {
        // number of elements allocated to each process
        send_counts[i] = quotient + (i < remainder ? 1 : 0);
    }

    displacement[0] = 0;
    for (int i = 1; i < world_size; i++) {
        // displacement of each process
        displacement[i] = displacement[i - 1] + send_counts[i - 1];
    }

    int num_my_element = send_counts[rank];
    int my_element[quotient + 1];  // store elements of each process

    // distribute elements to each process
    MPI_Scatterv(elements, send_counts, displacement, MPI_INT, my_element, num_my_element, MPI_INT, 0, MPI_COMM_WORLD);
    odd_even_sort(my_element, num_my_element, rank, num_elements, last_p, MPI_COMM_WORLD);
    // collect result from each process
    MPI_Gatherv(
        my_element, num_my_element, MPI_INT, sorted_elements, send_counts, displacement, MPI_INT, 0, MPI_COMM_WORLD
    );

    /* TODO END */

    if (rank == 0) {  // record time (only executed in master process)
        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Student ID: "
                  << "119020059" << std::endl;
        std::cout << "Name: "
                  << "Xinyu Xie" << std::endl;
        std::cout << "Assignment 1, Parallel version" << std::endl;
        std::cout << "Run Time: " << time_span.count() << " seconds" << std::endl;
        std::cout << "Input Size: " << num_elements << std::endl;
        std::cout << "Process Number: " << world_size << std::endl;

        // print input array and output array
        // std::cout << "The input array before sorting is: " << std::endl;
        // for (int i = 0; i < num_elements; i++) {
        //     std::cout << elements[i] << ' ';
        // }
        // std::cout << std::endl;
        // std::cout << "The output array after sorting is: " << std::endl;
        // for (int i = 0; i < num_elements; i++) {
        //     std::cout << sorted_elements[i] << ' ';
        // }
        // std::cout << std::endl;
    }

    if (rank == 0) {  // write result to file (only executed in master process)
        std::ofstream output(argv[2] + std::string(".parallel.out"), std::ios_base::out);
        for (int i = 0; i < num_elements; i++) {
            output << sorted_elements[i] << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
