#include "sdfg/Dialect/SDFGOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "sdfg/Dialect/IDGenerator.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/FunctionImplementation.h"

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

SDFGNode SDFGNode::create(PatternRewriter &rewriter, Location loc, StringRef name, TypeRange inputs, TypeRange results) {
  auto nameAttr = rewriter.getStringAttr(name);
  auto functionType = FunctionType::get(rewriter.getContext(), inputs, results);
  SDFGNode sdfg = rewriter.create<SDFGNode>(loc, nameAttr, TypeAttr::get(functionType), ArrayAttr(), ArrayAttr());
  Block *body = new Block();
  for (Type ty : inputs)
    body->addArgument(ty, loc);
  sdfg.getRegion().push_back(body);
  return sdfg;
}

Block::BlockArgListType SDFGNode::getFunctionArgs() {
  return getBody().getArguments();
}

// Custom parsing and printing for named arguments
ParseResult SDFGNode::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> inputs, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(inputs, results);
  };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void SDFGNode::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                           getFunctionTypeAttrName(),
                                           getArgAttrsAttrName(),
                                           getResAttrsAttrName());
}

mlir::Region* SDFGNode::getCallableRegion() const {
    auto &body = const_cast<SDFGNode *>(this)->getBody();
    if (body.empty())
        return nullptr;
    return &body;
}

mlir::ArrayRef<mlir::Type> SDFGNode::getArgumentTypes() const {
    static thread_local llvm::SmallVector<mlir::Type, 8> storage;
    storage.clear();
    auto funcTy = const_cast<SDFGNode *>(this)->getFunctionType();
    for (auto t : funcTy.getInputs())
        storage.push_back(t);
    return storage;
}
mlir::ArrayRef<mlir::Type> SDFGNode::getResultTypes() const {
    static thread_local llvm::SmallVector<mlir::Type, 8> storage;
    storage.clear();
    auto funcTy = const_cast<SDFGNode *>(this)->getFunctionType();
    for (auto t : funcTy.getResults())
        storage.push_back(t);
    return storage;
}
