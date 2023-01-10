include_directories(include)

set(headers include/graph.hpp
            include/min_cut.hpp)

set(sources src/main.cpp
            src/graph.cpp
            src/min_cut.cpp)

add_executable( Exercise5 ${sources} ${headers})
target_link_libraries( Exercise5 ${OpenCV_LIBS} )
set_property(TARGET Exercise5 PROPERTY CXX_STANDARD 17)