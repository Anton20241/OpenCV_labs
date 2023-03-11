cmake_minimum_required(VERSION 3.2)
project(5)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED )
add_executable(5 main.cpp cvDirectory.cpp cvDirectory.h)
target_link_libraries(5 ${OpenCV_LIBS} )
target_include_directories (5 PRIVATE aruco_calib/)
