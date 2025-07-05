#include "sdfg/Dialect/SDFGOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "sdfg/Dialect/IDGenerator.h"

using namespace mlir;
using namespace mlir::sdfg;

// Include generated operation definitions
#define GET_OP_CLASSES
#include "sdfg/Dialect/SDFGOps.cpp.inc"

// Include generated operation implementations
#define GET_OP_DEFS
#include "sdfg/Dialect/SDFGOps.cpp.inc"

// SDFGNodeOp

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification logic goes here (currently a no-op)
  return success();
}

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc,
                          unsigned num_args, TypeRange args) {
  // Generate a unique SDFG name from the ID.
  std::string name = "sdfg_" + std::to_string(utils::generateID());
  auto nameAttr = rewriter.getStringAttr(name);
  auto numArgsAttr = rewriter.getI32IntegerAttr(num_args);

  // Create the op using the rewriter (safe and RAII-compliant).
  SDFGNode sdfg = rewriter.create<SDFGNode>(loc, nameAttr, numArgsAttr);

  // Create the region block with the desired arguments.
  Block *body = new Block();
  for (Type ty : args)
    body->addArgument(ty, loc);
  sdfg.getRegion().push_back(body);

  return sdfg;
}

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc) {
  return create(rewriter, loc, 0, {});
}

Block::BlockArgListType SDFGNode::getArgs() {
  return getBody().getArguments().take_front(getNumArgs());
}

TypeRange SDFGNode::getArgTypes() {
  SmallVector<Type> types;
  for (BlockArgument arg : getArgs())
    types.push_back(arg.getType());
  return types;
}
