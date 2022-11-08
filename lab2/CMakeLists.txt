include_directories(include)

set(headers include/gamma_correction.hpp
            include/sobel_filter.hpp
            include/hist_equalization.hpp)

set(sources src/main.cpp
            src/gamma_correction.cpp
            src/sobel_filter.cpp
            src/hist_equalization.cpp)

add_executable( Exercise2 ${sources} ${headers})
target_link_libraries( Exercise2 ${OpenCV_LIBS} )
set_property(TARGET Exercise2 PROPERTY CXX_STANDARD 17)