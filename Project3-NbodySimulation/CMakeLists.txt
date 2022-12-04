cmake_minimum_required(VERSION 3.14)

if (APPLE)
	project(NBody LANGUAGES CXX)
else()
	project(NBody LANGUAGES CXX CUDA)
endif()

add_subdirectory(src)