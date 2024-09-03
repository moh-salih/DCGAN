cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(app LANGUAGES CXX C)

set(CMAKE_C_STANDARD    17)
set(CMAKE_CXX_STANDARD  20)
# set(CMAKE_CUDA_STANDARD 20)

if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
  set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
  set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

set(CMAKE_CUDA_FLAGS "-std=c++20")


# Find required dependencies.
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Project source files to include in the executable.
file(GLOB PROJECT_SOURCE_FILES "src/*.cpp")
list(APPEND PROJECT_SOURCE_FILES main.cpp)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_FILES})


# Specify include directories for the project.
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

# Link to the required dependencies.
target_link_libraries(${PROJECT_NAME}  PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES})