cmake_minimum_required( VERSION 3.22.1 )
project( "facedetector" )

add_library(
        ${CMAKE_PROJECT_NAME}
        SHARED
        src/lib.cpp
        src/face_detector.cpp)

target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        android
        log
)