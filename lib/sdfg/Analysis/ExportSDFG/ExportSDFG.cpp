#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/serializer/json_serializer.h"

namespace {

std::unique_ptr<sdfg::types::IType> mlir_type_to_sdfg_type(mlir::Type type) {
  // Handle scalar integer types (including MLIR index type)
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    switch (intType.getWidth()) {
    case 1:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Bool);
    case 8:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Int8);
    case 16:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Int16);
    case 32:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Int32);
    case 64:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Int64);
    case 128:
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Int128);
    default:
      break;
    }
  }
  if (mlir::isa<mlir::IndexType>(type)) {
    // Treat index type as signed 64-bit integer in the SDFG type system.
    return std::make_unique<sdfg::types::Scalar>(
      sdfg::types::StorageType_CPU_Stack,
      0,
      "",
      sdfg::types::PrimitiveType::Int64);
  }

  // Handle scalar floating-point types
  if (auto fpType = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (fpType.isF16())
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Half);
    if (fpType.isBF16())
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::BFloat);
    if (fpType.isF32())
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Float);
    if (fpType.isF64())
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::Double);
    if (fpType.isF128())
      return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        sdfg::types::PrimitiveType::FP128);
  }

  // Handle SDFG one-dimensional array type
  if (auto arrayType = mlir::dyn_cast<mlir::sdfg::ArrayType>(type)) {
    auto elementType = mlir_type_to_sdfg_type(arrayType.getElementType());
    auto numElements = sdfg::symbolic::integer(arrayType.getSize());

    // The SDFG runtime array type stores the element type by value and the
    // symbolic length of the array.
    return std::make_unique<sdfg::types::Array>(
        sdfg::types::StorageType_CPU_Stack,
        0,
        "",
        *elementType, numElements);
  }

  throw std::runtime_error("Unsupported type");
}

struct ExportSDFGPass : public mlir::sdfg::analysis::ExportSDFGPassBase<ExportSDFGPass> {
  void visit_alloca(sdfg::builder::StructuredSDFGBuilder& builder,
                    mlir::sdfg::AllocaOp allocaOp) {
    // MLIR operations do not expose an identifier string suitable for
    // naming containers directly.  Generate a deterministic placeholder
    // name instead.
    static unsigned allocaCounter = 0;
    std::string name = "alloca" + std::to_string(allocaCounter++);

    auto sdfg_type = mlir_type_to_sdfg_type(allocaOp.getType());
    builder.add_container(name, *sdfg_type, true);
  }

  void visit_library_node(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    llvm::outs() << "LibraryNode\n";
  }

  void visit_return(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::ReturnOp returnOp) {
    // Do nothing
    return;
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module.walk([&](mlir::sdfg::SDFGNode sdfgNode) {
      std::string sdfgName;
      if (auto symAttr = sdfgNode->getAttrOfType<mlir::StringAttr>("sym_name"))
        sdfgName = symAttr.getValue();
      else
        sdfgName = "_unnamed_";

      sdfg::builder::StructuredSDFGBuilder builder(sdfgName, sdfg::FunctionType_CPU);

      // Visit arguments of the SDFG function.
      auto funcTypeAttr = sdfgNode->getAttrOfType<mlir::TypeAttr>("function_type");
      if (funcTypeAttr) {
        auto funcType = mlir::cast<mlir::FunctionType>(funcTypeAttr.getValue());

        unsigned argIdx = 0;
        for (auto argType : funcType.getInputs()) {
          std::string name = "arg" + std::to_string(argIdx++);
          auto sdfg_type = mlir_type_to_sdfg_type(argType);
          builder.add_container(name, *sdfg_type, true);
        }
      }

      // Visit all operations in the SDFG body region.
      sdfgNode.getBody().walk([&](mlir::Operation *innerOp) {
        if (auto allocaOp = dyn_cast<mlir::sdfg::AllocaOp>(innerOp)) {
          visit_alloca(builder, allocaOp);
        } else if (auto libraryNodeOp = dyn_cast<mlir::sdfg::LibraryNodeOp>(innerOp)) {
          visit_library_node(builder, libraryNodeOp);
        } else if (auto returnOp = dyn_cast<mlir::sdfg::ReturnOp>(innerOp)) {
          visit_return(builder, returnOp);
        } else {
          throw std::runtime_error("Unsupported operation: " +
                                   innerOp->getName().getStringRef().str());
        }
      });

      // Finish SDFG
      auto sdfg = builder.move();

      // Serialize SDFG to JSON
      sdfg::serializer::JSONSerializer serializer;
      auto j = serializer.serialize(sdfg);
      std::filesystem::path sdfgPath = sdfgName + ".json";

      std::ofstream ofs(sdfgPath);
      if (!ofs.is_open()) {
          throw std::runtime_error("Failed to open file: " + sdfgPath.string());
      }
      ofs << j.dump(2);
      ofs.close();

    });
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> mlir::sdfg::analysis::createExportSDFGPass() {
  return std::make_unique<ExportSDFGPass>();
} 