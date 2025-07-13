#include "sdfg/Conversion/TorchToSDFG/PassDetail.h"
#include "sdfg/Conversion/TorchToSDFG/Passes.h"
#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOps.h"
#include "sdfg/Dialect/SDFGTypes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;
using namespace mlir::torch::Torch;

// Helper that recursively converts a `!torch.vtensor` type with fully static
// shape and known dtype to a nested `!sdfg.array` type.  If the conversion is
// not possible (e.g., dynamic sizes or unknown dtype), the original type is
// returned unchanged.
static mlir::Type convertVTensorToArray(mlir::Type ty) {
  auto *ctx = ty.getContext();
  if (auto vtTy = ty.dyn_cast<ValueTensorType>()) {
    // Only handle the fully-static case with a known dtype for now.
    if (!vtTy.hasSizes() || !vtTy.hasDtype())
      return ty;

    llvm::ArrayRef<int64_t> sizes = *vtTy.getOptionalSizes();
    for (int64_t sz : sizes)
      if (sz == mlir::ShapedType::kDynamic)
        return ty; // Dynamic dim – bail out.

    mlir::Type elemTy = vtTy.getDtype();
    // Build innermost-to-outermost nested array types.
    for (auto it = sizes.rbegin(), e = sizes.rend(); it != e; ++it) {
      elemTy = mlir::sdfg::ArrayType::get(ctx, static_cast<uint64_t>(*it), elemTy);
    }
    return elemTy;
  }
  return ty;
}

// Convert all vtensor types in the given module to nested sdfg.array types.
static void convertModuleVTensorsToArrays(mlir::ModuleOp module) {
  mlir::MLIRContext *ctx = module.getContext();

  // 1. Convert block argument types.
  module.walk([&](mlir::Block *block) {
    for (mlir::BlockArgument arg : block->getArguments()) {
      mlir::Type newTy = convertVTensorToArray(arg.getType());
      if (newTy != arg.getType())
        arg.setType(newTy);
    }
  });

  // 2. Convert operation result types.
  module.walk([&](mlir::Operation *op) {
    for (mlir::Value res : op->getResults()) {
      mlir::Type newTy = convertVTensorToArray(res.getType());
      if (newTy != res.getType())
        res.setType(newTy);
    }

    // Additionally, patch function-type attributes on sdfg.sdfg ops.
    if (auto sdfgOp = llvm::dyn_cast<mlir::sdfg::SDFGNode>(op)) {
      auto funcTyAttr = sdfgOp->getAttr("function_type").dyn_cast_or_null<mlir::TypeAttr>();
      if (!funcTyAttr)
        return;
      auto funcTy = funcTyAttr.getValue().cast<mlir::FunctionType>();
      llvm::SmallVector<mlir::Type> newInputs, newResults;
      newInputs.reserve(funcTy.getNumInputs());
      newResults.reserve(funcTy.getNumResults());
      bool changed = false;
      for (mlir::Type t : funcTy.getInputs()) {
        mlir::Type nt = convertVTensorToArray(t);
        newInputs.push_back(nt);
        changed |= (nt != t);
      }
      for (mlir::Type t : funcTy.getResults()) {
        mlir::Type nt = convertVTensorToArray(t);
        newResults.push_back(nt);
        changed |= (nt != t);
      }
      if (changed) {
        auto newFuncTy = mlir::FunctionType::get(ctx, newInputs, newResults);
        sdfgOp->setAttr("function_type", mlir::TypeAttr::get(newFuncTy));
      }
    }
  });
}

namespace {

// Lower `torch.operator "onnx.Constant"` to `sdfg.alloca` while forwarding the
// constant payload via a `value` attribute.
struct TorchConstantToAllocaPattern : public RewritePattern {
  TorchConstantToAllocaPattern(MLIRContext *context)
      : RewritePattern("torch.operator", /*benefit=*/2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // We are only interested in `torch.operator "onnx.Constant"`.
    if (!op->getName().getStringRef().starts_with("torch.operator"))
      return failure();

    // Extract the operator name (e.g., "onnx.Constant").  This follows the
    // same logic as the library-node pattern below.
    std::string opName;
    if (auto attr = op->getAttrOfType<StringAttr>("name")) {
      opName = attr.getValue().str();
    } else {
      auto opStr = op->getName().getStringRef().str();
      auto quotePos = opStr.find('"');
      if (quotePos != std::string::npos) {
        auto endQuote = opStr.find('"', quotePos + 1);
        if (endQuote != std::string::npos)
          opName = opStr.substr(quotePos + 1, endQuote - quotePos - 1);
      }
    }

    if (opName != "onnx.Constant")
      return failure();

    // Acquire the constant payload attribute (typically a DenseResourceAttr).
    Attribute valueAttr = op->getAttr("torch.onnx.value");

    if (!valueAttr)
      return failure();

    // Generate the `sdfg.alloca` op producing the same type as the constant.
    Type resultType = op->getResult(0).getType();

    auto alloca = rewriter.create<sdfg::AllocaOp>(op->getLoc(), resultType, valueAttr);

    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

struct TorchOperatorToLibraryNodePattern : public RewritePattern {
  TorchOperatorToLibraryNodePattern(MLIRContext *context)
      : RewritePattern("torch.operator", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Only match torch.operator ops
    if (!op->getName().getStringRef().starts_with("torch.operator"))
      return failure();

    // Get the operator name, e.g., "onnx.Conv"
    std::string opName;
    if (auto attr = op->getAttrOfType<StringAttr>("name")) {
      opName = attr.getValue().str();
    } else {
      // Try to parse from the op's custom format: torch.operator "onnx.Conv"
      auto opStr = op->getName().getStringRef().str();
      auto quotePos = opStr.find('"');
      if (quotePos != std::string::npos) {
        auto endQuote = opStr.find('"', quotePos + 1);
        if (endQuote != std::string::npos)
          opName = opStr.substr(quotePos + 1, endQuote - quotePos - 1);
      }
    }

    // If this is not a onnx. operator, do not lower
    if (opName.find("onnx.") == std::string::npos)
      return failure();

    // If this is a torch.operator "onnx.Constant", do not lower
    if (opName == "onnx.Constant")
      return failure();

    // drop onnx. prefix
    opName = opName.substr(strlen("onnx."));

    // Lower to sdfg.library_node
    SmallVector<Value> operands;
    for (auto operand : op->getOperands())
      operands.push_back(operand);

    // The new library_node expects exactly one result. Bail out if the source
    // operation produces a different number of results.  Multi-result
    // `torch.operator` cases need to be handled separately.
    if (op->getNumResults() != 1)
      return failure();

    // Extract relevant attributes from the torch operator (excluding the
    // builtin "name" attribute and stripping the "torch.onnx." prefix).
    SmallVector<NamedAttribute> attributes;
    for (auto attr : op->getAttrs()) {
      if (attr.getName() == "name")
        continue;

      auto nameStr = attr.getName().str();
      StringRef nameRef = nameStr;
      if (nameRef.starts_with("torch.onnx.")) {
        nameRef = nameRef.drop_front(strlen("torch.onnx."));
      }

      attributes.push_back(NamedAttribute(
          mlir::StringAttr::get(op->getContext(), nameRef), attr.getValue()));
    }

    // Use the opName as the code for the library node. The builder now takes
    // the single result type followed by the code attribute and operand list.
    auto libraryNode = rewriter.create<LibraryNodeOp>(
        op->getLoc(),
        op->getResult(0).getType(),
        rewriter.getStringAttr(opName),
        operands);

    // Add the extracted attributes to the library node
    for (auto attr : attributes) {
      libraryNode->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(op, libraryNode.getResult());
    return success();
  }
};

struct TorchToSDFGPass : public sdfg::conversion::TorchToSDFGPassBase<TorchToSDFGPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // 1. Lower torch.operator operations to sdfg.library_node.
    RewritePatternSet patterns(ctx);
    patterns.add<TorchConstantToAllocaPattern, TorchOperatorToLibraryNodePattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // 2. Convert func.func operations to sdfg.sdfg (and terminators).
    //    See original implementation above.
    for (auto funcOp : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      // Prepare a rewriter anchored at the current func.func op.
      PatternRewriter rewriter(ctx);
      rewriter.setInsertionPoint(funcOp);

      // Create the new sdfg.sdfg operation with the same signature and name.
      auto funcType = funcOp.getFunctionType();
      auto nameAttr = rewriter.getStringAttr(funcOp.getName());
      auto typeAttr = TypeAttr::get(funcType);

      // The SDFG op takes optional argument/result attribute arrays.  Forward
      // them if they exist.
      ArrayAttr argAttrs = funcOp.getAllArgAttrs();
      ArrayAttr resAttrs = funcOp.getAllResultAttrs();

      auto sdfgOp = rewriter.create<SDFGNode>(funcOp.getLoc(), nameAttr, typeAttr,
                                              argAttrs ? argAttrs : ArrayAttr(),
                                              resAttrs ? resAttrs : ArrayAttr());

      // Move the entire region body from the original function into the SDFG
      // op.  First, remove the automatically inserted empty block in the SDFG
      // op so we can splice the original block(s) directly.
      {
        Region &sdfgRegion = sdfgOp.getBody();
        if (!sdfgRegion.empty())
          rewriter.eraseBlock(&sdfgRegion.front());
      }

      // Inline the region using the rewriter helper to maintain bookkeeping.
      rewriter.inlineRegionBefore(funcOp.getBody(), sdfgOp.getBody(),
                                  sdfgOp.getBody().end());

      // Convert any func.return terminators inside the newly inlined region.
      sdfgOp.walk([&](func::ReturnOp retOp) {
        rewriter.setInsertionPoint(retOp);
        rewriter.replaceOpWithNewOp<sdfg::ReturnOp>(retOp, retOp.getOperands());
      });

      // Finally erase the original function.
      rewriter.eraseOp(funcOp);
    }

    // 3. Finally, convert all vtensor types to sdfg.array types.
    convertModuleVTensorsToArrays(module);
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::sdfg::conversion::createTorchToSDFGPass() {
  return std::make_unique<TorchToSDFGPass>();
} 