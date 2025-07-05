#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOps.h"

using namespace mlir;
using namespace mlir::sdfg;

//===----------------------------------------------------------------------===//
// SDFG dialect.
//===----------------------------------------------------------------------===//

void SDFGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "sdfg/Dialect/SDFGOps.cpp.inc"
      >();
}
