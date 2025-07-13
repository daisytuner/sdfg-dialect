cmake_minimum_required(VERSION 3.13.4)
project(sdfg-dialect LANGUAGES CXX C)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add IREE to find MLIR, LLVM, Torch-MLIR
add_subdirectory(3rdParty/iree)

# Set up LLVM and MLIR CMake directories after adding IREE
set(LLVM_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/iree/third_party/llvm-project/llvm/cmake/modules)
set(MLIR_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/iree/third_party/llvm-project/mlir/cmake/modules)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Set LLVM_TOOLS_BINARY_DIR to ensure FileCheck and other tools are found by lit tests
set(LLVM_TOOLS_BINARY_DIR ${LLVM_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Set up MLIR include directories properly
set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/iree/third_party/llvm-project/mlir)
set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_GENERATED_INCLUDE_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(sanitizers)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(sdfg-opt)
add_subdirectory(sdfg-export)
