include_directories(include)

set(headers include/median_filter.hpp
            include/connected_component.hpp)

set(sources src/main.cpp
            src/median_filter.cpp
            src/connected_component.cpp
            src/flood_filling.cpp)

add_executable( Exercise3 ${sources} ${headers})
target_link_libraries( Exercise3 ${OpenCV_LIBS} )
set_property(TARGET Exercise3 PROPERTY CXX_STANDARD 17)