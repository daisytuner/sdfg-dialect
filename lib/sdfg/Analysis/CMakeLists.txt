if(NOT TARGET sdfglib::sdfglib)
    find_package(sdfglib CONFIG REQUIRED)
endif()

add_mlir_dialect_library(
  ExportSDFG
  ExportSDFG/Nodes.cpp
  ExportSDFG/ExportSDFG.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sdfg/Analysis/ExportSDFG
  DEPENDS
  MLIRExportSDFGPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSDFG
  sdfglib::sdfglib
)
target_compile_options(ExportSDFG PRIVATE -w)
