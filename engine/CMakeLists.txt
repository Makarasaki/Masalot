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


cmake_minimum_required (VERSION 3.25.2)
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

# Find cuDNN
find_library(CUDNN_LIB NAMES cudnn HINTS /usr/local/cuda/lib64)
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS /usr/local/cuda/include)

if(CUDNN_LIB AND CUDNN_INCLUDE_DIR)
    message(STATUS "Found cuDNN: ${CUDNN_LIB}")
    include_directories(${CUDNN_INCLUDE_DIR})
    set(USE_CUDNN ON CACHE BOOL "Use cuDNN")
else()
    message(WARNING "cuDNN not found!")
endif()

# Set cuDNN configuration
set(CAFFE2_USE_CUDNN ON)
if(CAFFE2_USE_CUDNN)
    if(USE_STATIC_CUDNN)
        set(CUDNN_STATIC ON CACHE BOOL "")
    else()
        set(CUDNN_STATIC OFF CACHE BOOL "")
    endif()

    # Check if CUDNN library was found
    if(NOT CUDNN_LIB)
        message(WARNING "Cannot find cuDNN library. Turning the option off")
        set(CAFFE2_USE_CUDNN OFF)
    else()
        # Ensure the required version of cuDNN is met
        execute_process(COMMAND "${CUDNN_LIB}" --version OUTPUT_VARIABLE CUDNN_VERSION)
        if(CUDNN_VERSION VERSION_LESS "8.0.0")
            # message(FATAL_ERROR "PyTorch requires cuDNN 8 and above.")
        endif()
    endif()
else()
    message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
endif()

# Find libtorch
find_package(Torch REQUIRED)

# Specify the CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# Include directories for headers
# include_directories(${PROJECT_SOURCE_DIR}/include)

# Add your source files and set the output name to "Masalot"
add_executable(
    Masalot
    src/data_preparation.cpp
    ../training/src/chessnet.cpp
    # giga/Gigantua.cpp
    src/evaluate.cpp
    src/main.cpp
)

target_include_directories(
    Masalot PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/giga
    ${CUDNN_INCLUDE_DIR}
    # Add other include directories here
    ${PROJECT_SOURCE_DIR}/../training/include
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
)

# Set the required flags for linking libtorch
set_property(TARGET Masalot PROPERTY CXX_STANDARD 20)

# Required to suppress RPath errors
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Additional linker flags for libtorch
target_compile_features(Masalot PRIVATE cxx_std_20)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(Masalot PRIVATE -march=native -mbmi -mbmi2)
endif()

# Add this to ~/.zshrc
# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
# export PATH=$PATH:$CUDA_HOME/bin