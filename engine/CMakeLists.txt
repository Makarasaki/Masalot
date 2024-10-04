# cmake_minimum_required(VERSION 3.21)
# project(Masalot_engine)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "-Wno-deprecated")

# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#     set(CMAKE_CXX_FLAGS "-fconstexpr-steps=900000000")
# else()
#     set(CMAKE_CXX_FLAGS "-fconstexpr-ops-limit=900000000")
# endif()

# add_executable(Masalot include/move_gen.h src/main.cpp)



cmake_minimum_required(VERSION 3.21)
project(Masalot_engine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wno-deprecated")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-fconstexpr-steps=900000000")
else()
    set(CMAKE_CXX_FLAGS "-fconstexpr-ops-limit=900000000")
endif()

# Add the path to your libtorch installation (adjust the path as necessary)
set(CMAKE_PREFIX_PATH "/mnt/c/MAKS_STUDIA/MGR/test/pytorch_training/libtorch")

# Ensure the binary can find the necessary libtorch libraries at runtime
set(CMAKE_INSTALL_RPATH "/mnt/c/MAKS_STUDIA/MGR/test/pytorch_training/libtorch/lib")

# Find libtorch
find_package(Torch REQUIRED)

# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add your source files and set the output name to "training"
add_executable(
    Masalot
    src/data_preparation.cpp
    src/chessnet.cpp
    src/evaluate.cpp
    src/main.cpp
)

# Precompiled headers
target_precompile_headers(Masalot PRIVATE include/pch.h)

# Link libtorch and sqlite3 to your executable
target_link_libraries(Masalot "${TORCH_LIBRARIES}")

# Set the required flags for linking libtorch
set_property(TARGET Masalot PROPERTY CXX_STANDARD 20)

