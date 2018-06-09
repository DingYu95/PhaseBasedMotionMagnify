cmake_minimum_required(VERSION 3.0)
include(CMakeToolsHelpers OPTIONAL)
project(PhaseMag)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.2)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(WITH_CUDA ON)

set(SOURCE_FILES main.cpp ./src/preProcessor.cpp
                          ./src/spatialPyr.cpp
                          ./src/temporalFilter.cpp
                          ./src/magnifier.cpp)

find_package(OpenCV 2 REQUIRED)

include_directories(${root_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
