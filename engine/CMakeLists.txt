cmake_minimum_required(VERSION 3.25.2)
project(Masalot_engine)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wno-deprecated")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    set(CMAKE_CXX_FLAGS "-fconstexpr-steps=900000000")
else()
    set(CMAKE_CXX_FLAGS "-fconstexpr-ops-limit=900000000")
endif()

# Add the path to your libtorch installation (adjust the path as necessary)
set(CMAKE_PREFIX_PATH "/mnt/c/Maks/libtorch;/usr/local/cuda/lib64")

# Ensure the binary can find the necessary libtorch libraries at runtime
set(CMAKE_INSTALL_RPATH "/mnt/c/Maks/libtorch/lib")

# Find and include cuDNN
find_library(CUDNN_LIB NAMES cudnn HINTS /usr/local/cuda/lib64)
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS /usr/local/cuda/include)

if(CUDNN_LIB AND CUDNN_INCLUDE_DIR)
    message(STATUS "Found cuDNN: ${CUDNN_LIB}")
    include_directories(${CUDNN_INCLUDE_DIR})
    set(USE_CUDNN ON CACHE BOOL "Use cuDNN")
else()
    message(WARNING "cuDNN not found!")
endif()

set(CAFFE2_USE_CUDNN ON)
if(CAFFE2_USE_CUDNN)
    if(USE_STATIC_CUDNN)
        set(CUDNN_STATIC ON CACHE BOOL "")
    else()
        set(CUDNN_STATIC OFF CACHE BOOL "")
    endif()
else()
    message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
endif()

# Find libtorch
find_package(Torch REQUIRED)

# Find libcurl
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})  # So #include <curl/curl.h> works

# Specify the CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Add your source files
add_executable(
    Masalot
    src/data_preparation.cpp
    ../training/src/chessnet.cpp
    # giga/Gigantua.cpp
    # src/zorbist.cpp
    src/evaluate.cpp
    src/cloudDatabase.cpp
    src/main.cpp
)

# Include directories for headers
target_include_directories(
    Masalot PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/giga
    ${PROJECT_SOURCE_DIR}/../training/include
    # Already added CUDNN_INCLUDE_DIR above, but you can add here as well if needed
    # ${CUDNN_INCLUDE_DIR}
)

# Prefer pthreads for multithreading support
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Precompiled headers
target_precompile_headers(Masalot PRIVATE include/pch.h)

# Link libraries to your executable
target_link_libraries(Masalot
    "${CUDNN_LIB}"
    "${TORCH_LIBRARIES}"
    Threads::Threads
    ${CURL_LIBRARIES}     # <--- Link libcurl
)

# Set the required flags for linking libtorch
set_property(TARGET Masalot PROPERTY CXX_STANDARD 20)

# Required to suppress RPath errors
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Additional linker flags for libtorch
target_compile_features(Masalot PRIVATE cxx_std_20)

# Optional CPU-specific optimizations
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(Masalot PRIVATE -march=native -mbmi -mbmi2)
endif()
