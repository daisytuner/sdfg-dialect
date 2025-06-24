#ifndef SDFG_SDFGOPS_H
#define SDFG_SDFGOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "sdfg/SDFGOps.h.inc"

#endif // SDFG_SDFGOPS_H
