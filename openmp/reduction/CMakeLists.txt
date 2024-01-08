cmake_minimum_required(VERSION 3.9)
project(openmp_reduction)

set(EXE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/executable")
set(EXECUTABLE_OUTPUT_PATH  ${EXE_DIR})

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(openmp_reduction main.cpp)
target_link_libraries(openmp_reduction PUBLIC OpenMP::OpenMP_CXX)
