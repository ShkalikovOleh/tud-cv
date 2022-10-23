include_directories(include)

set(headers include/local_max.hpp)

set(sources src/main.cpp
            src/local_max.cpp)

add_executable( LocalMaxima ${sources} ${headers})
target_link_libraries( LocalMaxima ${OpenCV_LIBS} )