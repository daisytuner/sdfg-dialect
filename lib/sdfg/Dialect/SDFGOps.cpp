#include "sdfg/Dialect/SDFGOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"
#include "sdfg/Utils/IDGenerator.h"

using namespace mlir;
using namespace mlir::sdfg;

// Include generated operation definitions
#define GET_OP_CLASSES
#include "sdfg/Dialect/SDFGOps.cpp.inc"

// Include generated operation implementations
#define GET_OP_DEFS
#include "sdfg/Dialect/SDFGOps.cpp.inc"

LogicalResult SDFGNode::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verification logic goes here (currently a no-op)
  return success();
}

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc,
                          unsigned num_args, TypeRange args) {
  // Create SDFGNode directly with attributes.
  SDFGNode sdfg = rewriter.create<SDFGNode>(
      loc, utils::generateID(), num_args);

  // Prepare locations for block arguments.
  SmallVector<Location> argLocs(args.size(), loc);

  // Create entry block for the region.
  rewriter.createBlock(&sdfg.getRegion(), /*insertPt=*/{}, args, argLocs);

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

AllocSymbolOp AllocSymbolOp::create(PatternRewriter &rewriter, Location loc, StringRef sym) {
  return rewriter.create<AllocSymbolOp>(loc, rewriter.getStringAttr(sym));
}

AllocSymbolOp AllocSymbolOp::create(Location loc, StringRef sym) {
  OpBuilder builder(loc->getContext());
  return builder.create<AllocSymbolOp>(loc, builder.getStringAttr(sym));
}

