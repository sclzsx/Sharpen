cmake_minimum_required(VERSION 3.13)
project(mySharpen)

set(CMAKE_CXX_STANDARD 14)

add_executable(mySharpen main.cpp SharpenImpl.cpp SharpenImpl.hpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(mySharpen ${OpenCV_LIBS})