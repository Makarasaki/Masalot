cmake_minimum_required(VERSION 3.10)
project(ChessNN)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add the path to your libtorch installation (adjust the path as necessary)
set(CMAKE_PREFIX_PATH "/mnt/c/MAKS_STUDIA/MGR/test/pytorch_training/libtorch")

# Ensure the binary can find the necessary libtorch libraries at runtime
set(CMAKE_INSTALL_RPATH "/mnt/c/MAKS_STUDIA/MGR/test/pytorch_training/libtorch/lib")

# Find libtorch
find_package(Torch REQUIRED)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Include directories for headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add your source files and set the output name to "training"
add_executable(training
    src/main.cpp
    src/chessnet.cpp
    src/data_loader.cpp
)

# Precompiled headers
target_precompile_headers(training PRIVATE include/pch.h)

# Link libtorch and sqlite3 to your executable
target_link_libraries(training "${TORCH_LIBRARIES}" SQLite::SQLite3)

# Set the required flags for linking libtorch
set_property(TARGET training PROPERTY CXX_STANDARD 17)
