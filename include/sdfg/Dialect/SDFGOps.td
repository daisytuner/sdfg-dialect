#ifndef SDFG_OPS
#define SDFG_OPS

include "SDFGDialect.td"
include "mlir/IR/OpBase.td" 
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Nodes
//===----------------------------------------------------------------------===//

include "sdfg/Dialect/controlflow/sdfg.td"
include "sdfg/Dialect/controlflow/block.td"
include "sdfg/Dialect/dataflow/library_node.td"

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

#endif // SDFG_OPS
