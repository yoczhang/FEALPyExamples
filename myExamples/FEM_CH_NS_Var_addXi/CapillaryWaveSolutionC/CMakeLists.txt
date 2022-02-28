cmake_minimum_required(VERSION 2.8)
project(capillary_wave)

enable_language(Fortran)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

set(PROJECT_NAME capillary_wave_app)
add_executable(${PROJECT_NAME} 
    capillary_wave/capillary_wave.C 
    capillary_wave/dmath_util.C 
    capillary_wave/680.f
    capillary_wave/merrcz.f)
#target_link_libraries(${PROJECT_NAME} gsl gslcblas)
target_link_libraries(${PROJECT_NAME} "/usr/local/lib/libgsl.a")
target_link_libraries(${PROJECT_NAME} "/usr/local/lib/libgslcblas.a")
target_include_directories(${PROJECT_NAME} PRIVATE "/usr/local/include/")
