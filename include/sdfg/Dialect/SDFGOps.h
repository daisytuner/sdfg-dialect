#ifndef SDFG_SDFGOPS_H
#define SDFG_SDFGOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "sdfg/Dialect/SDFGOps.h.inc"

#endif // SDFG_SDFGOPS_H
