include_directories(include)

set(headers include/feature_detection.hpp
            include/pixel_classification.hpp)

set(sources src/main.cpp
            src/feature_detection.cpp
            src/pixel_classification.cpp)

add_executable( Exercise4 ${sources} ${headers})
target_link_libraries( Exercise4 ${OpenCV_LIBS} )
set_property(TARGET Exercise4 PROPERTY CXX_STANDARD 17)