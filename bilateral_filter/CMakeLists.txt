include_directories(include)

set(headers include/bilateral_filter.hpp include/utility.hpp)

set(sources src/main.cpp src/bilateral_filter.cpp)

add_executable( bilateral_filter ${sources} ${headers} )
target_link_libraries( bilateral_filter ${OpenCV_LIBS} )
