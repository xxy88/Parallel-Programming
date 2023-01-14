cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# specify debug and release flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
#set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
if (USER_DEFINED_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_DEFINED_CXX_FLAGS}")
endif()

include_directories("headers")

# add executables
add_executable(sequential "sequential.cpp" )
add_executable(pthread "pthread.cpp" )
add_executable(mpi "mpi.cpp" )
if (NOT APPLE)
    add_executable(openmp "openmp.cpp" )
    add_executable(mpiomp "mpiomp.cpp" )
    add_executable(cuda "cuda.cu" )

    # CUDA configurations
    set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES 61-real 75-real 86-real)
    target_include_directories(cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# link Pthreads
if (MSVC)
    find_package(PThreads4W REQUIRED)
	target_link_libraries(pthread PRIVATE PThreads4W::PThreads4W)
else ()
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
    find_package(Threads REQUIRED)
    target_link_libraries(pthread PRIVATE Threads::Threads)
endif()

# link MPI
find_package(MPI REQUIRED)
target_link_libraries(mpi PRIVATE MPI::MPI_CXX)
if (NOT APPLE)
    target_link_libraries(mpiomp PRIVATE MPI::MPI_CXX)
endif()

if (NOT APPLE)
    # link OpenMP
    find_package(OpenMP REQUIRED)
    target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)
	target_link_libraries(mpiomp PRIVATE OpenMP::OpenMP_CXX)
endif()

# link GUI
if(GUI)
    # find and link GLUT
    find_package(OpenGL REQUIRED)
    find_package(GLUT REQUIRED)

    if (APPLE)
        include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
        target_link_libraries(sequential PRIVATE ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
        target_link_libraries(pthread PRIVATE ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
        target_link_libraries(mpi PRIVATE ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
    else()
        target_link_libraries(sequential PRIVATE GLUT::GLUT)
        target_link_libraries(pthread PRIVATE GLUT::GLUT)
        target_link_libraries(mpi PRIVATE GLUT::GLUT)
        target_link_libraries(openmp PRIVATE GLUT::GLUT)
		target_link_libraries(mpiomp PRIVATE GLUT::GLUT)
        target_link_libraries(cuda PRIVATE GLUT::GLUT)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGUI")
	endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGUI")
endif()
