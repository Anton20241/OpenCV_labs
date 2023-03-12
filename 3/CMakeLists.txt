cmake_minimum_required(VERSION 3.2)
project(3)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )
add_executable(3 main.cpp LAB3/cvDirectory.cpp LAB3/cvDirectory.h)
target_link_libraries( 3 ${OpenCV_LIBS} )

target_include_directories (3 PRIVATE LAB3/)
