#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

void swap(int &x, int &y) {
    int temp = x;
    x = y;
    y = temp;
}

int main(int argc, char **argv) {
    int num_elements;  // number of elements to be sorted
    num_elements = atoi(argv[1]);  // convert command line argument to num_elements

    int elements[num_elements];  // store elements
    int sorted_elements[num_elements];  // store sorted elements

    std::ifstream input(argv[2]);
    int element;
    int i = 0;
    while (input >> element) {
        elements[i] = element;
        i++;
    }
    std::copy(elements, elements + num_elements, sorted_elements);

    std::cout << "actual number of elements:" << i << std::endl;

    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    t1 = std::chrono::high_resolution_clock::now();  // record time

    /* TODO BEGIN
        Implement sequential odd even transposition sort
        Code in this block is not a necessary.
        Replace it with your own code.
    */

    bool is_changed;
    while (true) {
        is_changed = false;
        for (size_t i = 0; i < num_elements - 1; i += 2) {
            if (sorted_elements[i + 1] < sorted_elements[i]) {
                swap(sorted_elements[i + 1], sorted_elements[i]);
                is_changed = true;
            }
        }
        for (size_t i = 1; i < num_elements - 1; i += 2) {
            if (sorted_elements[i] > sorted_elements[i + 1]) {
                swap(sorted_elements[i + 1], sorted_elements[i]);
                is_changed = true;
            }
        }
        if (is_changed == false) {
            break;
        }
    }

    /* TODO END */

    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Student ID: "
              << "119020059" << std::endl;  // replace it with your student id
    std::cout << "Name: "
              << "Xinyu Xie" << std::endl;  // replace it with your name
    std::cout << "Assignment 1, Sequential version" << std::endl;
    std::cout << "Run Time: " << time_span.count() << " seconds" << std::endl;
    std::cout << "Input Size: " << num_elements << std::endl;
    std::cout << "Process Number: " << 1 << std::endl;

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

    std::ofstream output(argv[2] + std::string(".seq.out"), std::ios_base::out);
    for (int i = 0; i < num_elements; i++) {
        output << sorted_elements[i] << std::endl;
    }

    return 0;
}
