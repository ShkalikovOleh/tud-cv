include_directories(include)

set(headers include/median_filter.hpp)

set(sources src/main.cpp
            src/median_filter.cpp)

add_executable( Exercise3 ${sources} ${headers})
target_link_libraries( Exercise3 ${OpenCV_LIBS} )
set_property(TARGET Exercise3 PROPERTY CXX_STANDARD 17)