#include "sdfg/Conversion/LinalgToSDFG/PassDetail.h"
#include "sdfg/Conversion/LinalgToSDFG/Passes.h"
#include "sdfg/Dialect/SDFGDialect.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace sdfg;
using namespace conversion;

//===----------------------------------------------------------------------===//
// Target & Type Converter
//===----------------------------------------------------------------------===//

/// Defines the target to convert to.
struct SDFGTarget : public ConversionTarget {
  SDFGTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    // Mark operations as illegal
    addIllegalOp<linalg::MatmulOp>();
    
    // Every other operation is legal (best effort)
    markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
  }
};

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert linalg.matmul to sdfg.library_node
struct MatmulToLibraryNodePattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    // Get the operands and output
    Value lhs = op.getInputs()[0];
    Value rhs = op.getInputs()[1];
    Value output = op.getOutputs()[0];
    
    // Create the library node with "gemm" as the code
    // Pass lhs, rhs as inputs and output as an output argument
    SmallVector<Value> inputs = {lhs, rhs};
    SmallVector<Value> outputs = {output};
    auto libraryNode = rewriter.create<LibraryNodeOp>(
        op.getLoc(), 
        TypeRange{output.getType()},
        rewriter.getStringAttr("math.gemm"),
        inputs,
        outputs);
    
    // Replace the matmul operation with the library node result
    rewriter.replaceOp(op, libraryNode.getResults());
    
    return success();
  }
};



//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Registers all the patterns above in a RewritePatternSet.
void populateLinalgToSDFGConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MatmulToLibraryNodePattern>(patterns.getContext());
}

namespace {
struct LinalgToSDFGPass
    : public sdfg::conversion::LinalgToSDFGPassBase<LinalgToSDFGPass> {
  LinalgToSDFGPass() = default;

  void runOnOperation() override;
};
} // namespace

/// Runs the pass on the top-level module operation.
void LinalgToSDFGPass::runOnOperation() {
  ModuleOp module = getOperation();

  SDFGTarget target(getContext());

  RewritePatternSet patterns(&getContext());
  populateLinalgToSDFGConversionPatterns(patterns);

  if (applyFullConversion(module, target, std::move(patterns)).failed())
    signalPassFailure();
}

/// Returns a unique pointer to this pass.
std::unique_ptr<Pass> conversion::createLinalgToSDFGPass() {
  return std::make_unique<LinalgToSDFGPass>();
}
