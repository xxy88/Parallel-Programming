cmake_minimum_required(VERSION 3.14)

if (APPLE)
	project(HeatSimulation LANGUAGES CXX)
else()
	project(HeatSimulation LANGUAGES CXX CUDA)
endif()

add_subdirectory(src)