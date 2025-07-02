#ifndef SDFG_OPS
#define SDFG_OPS

include "SDFGDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "sdfg/Dialect/nodes/sdfg.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

include "sdfg/Dialect/ops/symbol.td"

#endif // SDFG_OPS
