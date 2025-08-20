# Check if Google Test is already available, if not fetch it
if(NOT TARGET gtest AND NOT TARGET gtest_main)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

set(EXPORTSDFG_TEST_FILES
    test.cpp
    operations_test.cpp
    library_nodes/elementwise_unary_test.cpp
    library_nodes/elementwise_binary_test.cpp
)

add_executable(export_sdfg_test ${EXPORTSDFG_TEST_FILES})
target_include_directories(export_sdfg_test PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(export_sdfg_test PRIVATE
    gtest_main 
    ${dialect_libs}
    ${conversion_libs}
    MLIRSDFG
    ExportSDFG
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRAnalysis
    MLIRParser
    MLIRAsmParser
    MLIRInferTypeOpInterface
    MLIRDialectUtils
)

add_test(NAME export_sdfg_test COMMAND export_sdfg_test)

include(GoogleTest)
gtest_discover_tests(export_sdfg_test)
