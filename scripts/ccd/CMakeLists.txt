cmake_minimum_required(VERSION 3.14)
project(fcl_ccd_check CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths to the local FCL build (cloned at ../../fcl)
set(FCL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../fcl")
set(FCL_INCLUDE_DIR "${FCL_SOURCE_DIR}/include")
set(FCL_BUILD_DIR "${FCL_SOURCE_DIR}/build")
set(FCL_BUILD_INCLUDE_DIR "${FCL_BUILD_DIR}/include")
set(FCL_LIBRARY_DIR "${FCL_BUILD_DIR}/lib")

find_library(FCL_LIBRARY fcl PATHS ${FCL_LIBRARY_DIR} REQUIRED)
find_library(CCD_LIBRARY ccd REQUIRED)

find_package(Eigen3 3.0.5 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(assimp REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(fcl_ccd_check fcl_ccd_check.cpp)

target_include_directories(fcl_ccd_check PRIVATE
    ${FCL_INCLUDE_DIR}
    ${FCL_BUILD_INCLUDE_DIR}
)

set(ASSIMP_TARGET assimp::assimp)
if(NOT TARGET ${ASSIMP_TARGET})
    set(ASSIMP_TARGET ${ASSIMP_LIBRARIES})
    target_include_directories(fcl_ccd_check PRIVATE ${ASSIMP_INCLUDE_DIRS})
endif()

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
target_compile_definitions(fcl_ccd_check PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT}")

target_link_libraries(fcl_ccd_check PRIVATE
    ${FCL_LIBRARY}
    ${CCD_LIBRARY}
    Eigen3::Eigen
    pinocchio::pinocchio
    yaml-cpp
    ${ASSIMP_TARGET}
)

target_compile_options(fcl_ccd_check PRIVATE -O3 -Wall -Wextra)

install(TARGETS fcl_ccd_check DESTINATION bin)
