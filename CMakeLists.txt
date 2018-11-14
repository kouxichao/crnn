
cmake_minimum_required(VERSION 3.8)

set (CMAKE_C_COMPILER aarch64-himix100-linux-gcc)#/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc)
set (CMAKE_CXX_COMPILER aarch64-himix100-linux-g++)#/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++)

project(demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -mcpu=cortex-a53+simd -mcpu=cortex-a53+fp")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/ncnn/)
FIND_LIBRARY(DLIB dlib ${CMAKE_SOURCE_DIR})
FIND_LIBRARY(ncnn_LIB ncnn ${CMAKE_SOURCE_DIR})#${CMAKE_SOURCE_DIR}/../../build-android-aarch64/src)

set(SRC_LIST interface_crnn.cpp)

add_library(vision STATIC ${SRC_LIST})

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

add_executable(demo_crnn demo_crnn.cpp)
target_link_libraries(demo_crnn vision ${ncnn_LIB} ${DLIB} -ldl -lpthread)

