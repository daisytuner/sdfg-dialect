add_mlir_dialect_library(MLIRSDFG
        SDFGDialect.cpp
        SDFGOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sdfg

        DEPENDS
        MLIRSDFGOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
