cmake_minimum_required(VERSION 3.0)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory(bilateral_filter)
add_subdirectory(lab1)