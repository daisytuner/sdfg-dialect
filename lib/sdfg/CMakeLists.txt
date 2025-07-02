add_mlir_dialect_library(MLIRSDFG
        Dialect/SDFGDialect.cpp
        Dialect/SDFGOps.cpp
        Utils/IDGenerator.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/sdfg

        DEPENDS
        MLIRSDFGOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
