cmake_minimum_required(VERSION 3.25)
project(4)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )
add_executable(4 main.cpp cvDirectory.cpp cvDirectory.h)
target_link_libraries( 4 ${OpenCV_LIBS} )