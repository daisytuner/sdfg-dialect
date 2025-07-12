#ifndef SDFG_OPS
#define SDFG_OPS

include "SDFGDialect.td"
include "mlir/IR/OpBase.td" 
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

include "sdfg/Dialect/sdfg.td"

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "sdfg/Dialect/nodes/library_node.td"
include "sdfg/Dialect/nodes/alloca.td"
include "sdfg/Dialect/nodes/return.td"

#endif // SDFG_OPS
