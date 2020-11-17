cmake_minimum_required(VERSION 3.14)
project(CV06)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d opencv_viz)
set(SRC cli.h colors.h timer.h aabb.cpp aabb.h camera.cpp camera.h cell.cpp cell.h intersection.cpp intersection.h matrix4x4.cpp matrix4x4.h ray.cpp ray.h stdafx.cpp stdafx.h vector3.cpp vector3.h volume.cpp volume.h)

add_executable(${PROJECT_NAME} ${SRC} main.cpp)

find_package(OpenMP REQUIRED)

if (OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
else ()
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif ()

