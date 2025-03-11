cmake_minimum_required(VERSION 3.10)
project(ChessNN)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the path to your libtorch installation (adjust the path as necessary)
set(CMAKE_PREFIX_PATH "/mnt/c/Maks/libtorch")
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

# cudnn interface
# static linking is handled by USE_STATIC_CUDNN environment variable
set(CAFFE2_USE_CUDNN ON)
if(CAFFE2_USE_CUDNN)
  if(USE_STATIC_CUDNN)
    set(CUDNN_STATIC ON CACHE BOOL "")
  else()
    set(CUDNN_STATIC OFF CACHE BOOL "")
  endif()

  find_package(CUDNN)

  if(NOT CUDNN_FOUND)
    message(WARNING
      "Cannot find cuDNN library. Turning the option off")
    set(CAFFE2_USE_CUDNN OFF)
  else()
    if(CUDNN_VERSION VERSION_LESS "8.0.0")
      message(FATAL_ERROR "PyTorch requires cuDNN 8 and above.")
    endif()
  endif()

  add_library(torch::cudnn INTERFACE IMPORTED)
  target_include_directories(torch::cudnn INTERFACE ${CUDNN_INCLUDE_PATH})
  if(CUDNN_STATIC AND NOT WIN32)
    target_link_options(torch::cudnn INTERFACE
        "-Wl,--exclude-libs,libcudnn_static.a")
  else()
    target_link_libraries(torch::cudnn INTERFACE ${CUDNN_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
endif()

# Find libtorch
find_package(Torch REQUIRED)

# Find SQLite3
find_package(SQLite3 REQUIRED)

# Include directories for your project
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add your source files and set the output name to "training"
add_executable(training
    src/main.cpp
    src/chessnet.cpp
    src/data_loader.cpp
)

# Ensure linking with pthreads (for multithreading support)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Precompiled headers (Optional but useful for large projects)
target_precompile_headers(training PRIVATE include/pch.h)

# Link libtorch, SQLite3, and pthreads to your executable
target_link_libraries(training 
    "${CUDNN_LIB}"
    "${TORCH_LIBRARIES}" 
    SQLite::SQLite3 
    Threads::Threads
)

# Set the required flags for linking libtorch
set_property(TARGET training PROPERTY CXX_STANDARD 17)

# Required to suppress RPath errors
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Additional linker flags for libtorch
target_compile_features(training PRIVATE cxx_std_17)

# Add this to ~/.zshrc
# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
# export PATH=$PATH:$CUDA_HOME/bin