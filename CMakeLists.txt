cmake_minimum_required(VERSION 3.0)

# set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(lab1)
add_subdirectory(lab2)
