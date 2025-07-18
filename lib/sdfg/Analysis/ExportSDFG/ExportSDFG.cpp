#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/metadata_node.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/visualizer/dot_visualizer.h>

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

// Helper to normalize SSA names so they are valid SDFG container identifiers.
static std::string normalize_name(std::string name) {
  std::replace(name.begin(), name.end(), '.', '_');
  std::replace(name.begin(), name.end(), ':', '_');
  std::replace(name.begin(), name.end(), '-', '_');
  std::replace(name.begin(), name.end(), '%', '_');
  std::replace(name.begin(), name.end(), '@', '_');
  std::replace(name.begin(), name.end(), '(', '_');
  std::replace(name.begin(), name.end(), ')', '_');
  std::replace(name.begin(), name.end(), '&', '_');
  std::replace(name.begin(), name.end(), '#', '_');

  // Remove a few characters entirely.
  auto eraseChars = {'<', '>', ' ', '*', ','};
  for (char c : eraseChars)
    name.erase(std::remove(name.begin(), name.end(), c), name.end());

  if (name == "badref")
    return "";
  if (name == "this")
    return "self";
  return "_" + name;
}

// Helper to print an MLIR value as operand (e.g., "%3" or "%arg0") and
// convert it into a normalized string usable as a container/access node name.
static std::string mlir_value_to_name(mlir::Value value) {
  std::string tmp;
  llvm::raw_string_ostream os(tmp);
  // Use a local printing scope to avoid the need for an AsmState managed by
  // the caller.
  value.printAsOperand(os, mlir::OpPrintingFlags().useLocalScope());
  os.flush();

  // Remove the leading '%' that is printed for SSA values.
  if (!tmp.empty() && tmp.front() == '%')
    tmp.erase(tmp.begin());

  return normalize_name(tmp);
}

void sdfg_array_to_subset(const sdfg::types::Array& array, sdfg::data_flow::Subset& begin_subset, sdfg::data_flow::Subset& end_subset) {
  auto& element_type = array.element_type();
  auto num_elements = array.num_elements();

  begin_subset.push_back(sdfg::symbolic::integer(0));
  end_subset.push_back(num_elements);

  if (element_type.type_id() == sdfg::types::TypeID::Scalar) {
    return;
  }
  sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(element_type), begin_subset, end_subset);
}

struct ExportSDFGPass : public mlir::sdfg::analysis::ExportSDFGPassBase<ExportSDFGPass> {
  void visit_alloca(sdfg::builder::StructuredSDFGBuilder& builder,
                    mlir::sdfg::AllocaOp allocaOp) {
    std::string name = mlir_value_to_name(allocaOp.getResult());

    auto sdfg_type = mlir_type_to_sdfg_type(allocaOp.getType());
    builder.add_container(name, *sdfg_type);
  }

  void visit_library_node(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // Add inputs
    std::vector<std::string> inputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> inputAccessNodes;
    for (auto arg : libraryNodeOp.getOperands()) {
      std::string input = mlir_value_to_name(arg);
      inputs.push_back(input);
      auto& access_node = builder.add_access(block, input);
      inputAccessNodes[input] = &access_node;
    }

    // Add outputs
    std::vector<std::string> outputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> outputAccessNodes;
    for (auto result : libraryNodeOp.getResults()) {
      std::string output = mlir_value_to_name(result);
      auto sdfg_type = mlir_type_to_sdfg_type(result.getType());
      builder.add_container(output, *sdfg_type);

      outputs.push_back(output);
      auto& access_node = builder.add_access(block, output);
      outputAccessNodes[output] = &access_node;
    }

    // Add operator to metadata
    std::unordered_map<std::string, std::string> metadata;
    std::string code = libraryNodeOp.getCode().str();
    metadata["frontend"] = "mlir";
    metadata["dialect"] = "torch-mlir";
    metadata["operator"] = code;

    // Add node attributes to metadata. Iterate over the raw operation's
    // attribute list (ArrayRef<NamedAttribute>) and stringify both key and
    // value so they can be stored in the MetadataNode.
    for (auto namedAttr : libraryNodeOp->getAttrs()) {
      std::string key = namedAttr.getName().str();

      // Convert the attribute value to a human-readable string. For string
      // attributes we can use the contained value directly. For all other
      // attribute kinds fall back to MLIR's generic printing.
      std::string value;
      if (auto strAttr = namedAttr.getValue().dyn_cast<mlir::StringAttr>()) {
        value = strAttr.getValue().str();
      } else {
        llvm::raw_string_ostream os(value);
        namedAttr.getValue().print(os);
      }

      metadata[key] = value;
    }

    auto& library_node = builder.add_library_node<sdfg::data_flow::MetadataNode>(block, sdfg::DebugInfo(), outputs, inputs, metadata);

    for (auto input : inputs) {
      auto inputAccessNode = inputAccessNodes[input];
      auto& input_type = sdfg.type(input);

      sdfg::data_flow::Subset begin_subset;
      sdfg::data_flow::Subset end_subset;
      if (input_type.type_id() == sdfg::types::TypeID::Array) {
        sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset, end_subset);
      } else {
        begin_subset.push_back(sdfg::symbolic::integer(0));
        end_subset.push_back(sdfg::symbolic::integer(0));
      }
      builder.add_memlet(block, *inputAccessNode, "void", library_node, input, begin_subset, end_subset);
    }

    for (auto output : outputs) {
      auto outputAccessNode = outputAccessNodes[output];
      auto& output_type = sdfg.type(output);

      sdfg::data_flow::Subset begin_subset;
      sdfg::data_flow::Subset end_subset;
      if (output_type.type_id() == sdfg::types::TypeID::Array) {
        sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(output_type), begin_subset, end_subset);
      } else {
        begin_subset.push_back(sdfg::symbolic::integer(0));
        end_subset.push_back(sdfg::symbolic::integer(0));
      }
      builder.add_memlet(block, library_node, output, *outputAccessNode, "void", begin_subset, end_subset);
    }
  }

  void visit_return(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::ReturnOp returnOp) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    builder.add_return(root);
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
          std::string name = mlir_value_to_name(sdfgNode.getArgument(argIdx++));
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

      // simplify CFG
      sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
      sdfg::passes::Pipeline cfg_simplifier("CFG Simplifier");
      cfg_simplifier.register_pass<sdfg::passes::BlockFusionPass>();
      cfg_simplifier.register_pass<sdfg::passes::DeadCFGElimination>();
      cfg_simplifier.run(builder, analysis_manager);

      // Finish SDFG
      auto sdfg = builder.move();

      sdfg::visualizer::DotVisualizer visualizer(*sdfg);
      visualizer.visualize();
      std::filesystem::path dotPath = sdfgName + ".dot";
      visualizer.writeToFile(*sdfg, &dotPath);

      // Serialize SDFG to JSON
      sdfg::serializer::JSONSerializer serializer;
      auto j = serializer.serialize(*sdfg);
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