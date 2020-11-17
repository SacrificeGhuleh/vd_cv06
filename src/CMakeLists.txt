cmake_minimum_required(VERSION 3.14)
project(CV06)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d opencv_viz)
set(SRC cli.h colors.h timer.h)

add_executable(${PROJECT_NAME} ${SRC} main.cpp)

find_package(OpenMP REQUIRED)

if (OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else ()
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif ()

