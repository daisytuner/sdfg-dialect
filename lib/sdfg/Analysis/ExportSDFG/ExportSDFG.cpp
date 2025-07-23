#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "sdfg/Analysis/ExportSDFG/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/metadata_node.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/visualizer/dot_visualizer.h>

namespace {

static std::vector<size_t> attribute_to_size_t_vector(mlir::Attribute attr) {
  std::vector<size_t> result;

  // Case 1: ArrayAttr of IntegerAttr
  if (auto arr = attr.dyn_cast<mlir::ArrayAttr>()) {
    result.reserve(arr.size());
    for (auto element : arr) {
      if (auto intAttr = element.dyn_cast<mlir::IntegerAttr>()) {
        // Use APInt helpers to avoid sign-ness assertion in getInt().
        result.push_back(static_cast<size_t>(intAttr.getValue().getZExtValue()));
      }
    }
    return result;
  }

  // Case 2: Dense int elements (e.g., DenseI64ArrayAttr, DenseElementsAttr of ints)
  if (auto dense = attr.dyn_cast<mlir::DenseIntElementsAttr>()) {
    result.reserve(static_cast<size_t>(dense.size()));
    for (auto value : dense.getValues<llvm::APInt>()) {
      result.push_back(static_cast<size_t>(value.getZExtValue()));
    }
    return result;
  }

  // Fallback: return empty vector if unsupported type.
  return result;
}

struct ExportSDFGPass : public mlir::sdfg::analysis::ExportSDFGPassBase<ExportSDFGPass> {
  void visit_alloca(sdfg::builder::StructuredSDFGBuilder& builder,
                    mlir::sdfg::AllocaOp allocaOp) {
    std::string name = sdfg::analysis::mlir_value_to_name(allocaOp.getResult());

    auto sdfg_type = sdfg::analysis::mlir_type_to_sdfg_type(allocaOp.getType());
    builder.add_container(name, *sdfg_type);
  }

  void visit_library_node(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    std::string code = libraryNodeOp.getCode().str();
    
    bool success = false;
    if (code == "Relu") {
      success = visit_relu(builder, libraryNodeOp);
    } else if (code == "MaxPool") {
      success = visit_maxpool(builder, libraryNodeOp);
    } else if (code == "Conv") {
      success = visit_conv(builder, libraryNodeOp);
    }

    if (!success) {
      visit_metadata_node(builder, libraryNodeOp);
    }
  }

  bool visit_maxpool(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    if (libraryNodeOp.getOperands().size() != 1) {
      return false;
    }
    if (libraryNodeOp.getResults().size() != 1) {
      return false;
    }

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // Define input
    auto input = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[0]);
    auto& input_node = builder.add_access(block, input);

    // Define output
    auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto sdfg_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(output, *sdfg_type);
    auto& output_node = builder.add_access(block, output);

    // Define attributes
    std::vector<size_t> kernel_shape;
    std::vector<size_t> pads;
    std::vector<size_t> strides;
    for (auto namedAttr : libraryNodeOp->getAttrs()) {
      auto attrName = namedAttr.getName().getValue();
      if (attrName == "kernel_shape") {
        kernel_shape = attribute_to_size_t_vector(namedAttr.getValue());
      } else if (attrName == "pads") {
        pads = attribute_to_size_t_vector(namedAttr.getValue());
      } else if (attrName == "strides") {
        strides = attribute_to_size_t_vector(namedAttr.getValue());
      }
    }

    auto& library_node = static_cast<sdfg::math::ml::MaxPoolNode&>(builder.add_library_node<sdfg::math::ml::MaxPoolNode>(block, sdfg::DebugInfo(), kernel_shape, pads, strides));

    // Add input memlet
    auto& input_type = sdfg.type(input);
    sdfg::data_flow::Subset begin_subset_in;
    sdfg::data_flow::Subset end_subset_in;
    if (input_type.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset_in, end_subset_in);
    } else {
      begin_subset_in.push_back(sdfg::symbolic::integer(0));
      end_subset_in.push_back(sdfg::symbolic::integer(0));
    }
    auto& iedge = builder.add_computational_memlet(block, input_node, library_node, "X", begin_subset_in, end_subset_in);

    // Add output memlet
    auto& output_type = sdfg.type(output);
    sdfg::data_flow::Subset begin_subset_out;
    sdfg::data_flow::Subset end_subset_out;
    if (output_type.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(output_type), begin_subset_out, end_subset_out);
    } else {
      begin_subset_out.push_back(sdfg::symbolic::integer(0));
      end_subset_out.push_back(sdfg::symbolic::integer(0));
    }
    auto& oedge = builder.add_computational_memlet(block, library_node, "Y", output_node, begin_subset_out, end_subset_out);

    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    if (!library_node.expand(builder, analysis_manager)) {
      builder.remove_memlet(block, iedge);
      builder.remove_memlet(block, oedge);
      builder.remove_node(block, library_node);
      builder.remove_node(block, input_node);
      builder.remove_node(block, output_node);
      return false;
    }

    return true;
  }

  bool visit_conv(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    // Expect either 2 (input + weight) or 3 (input + weight + bias) operands
    size_t numOperands = libraryNodeOp.getOperands().size();
    if (numOperands < 2 || numOperands > 3) {
      return false;
    }
    if (libraryNodeOp.getResults().size() != 1) {
      return false;
    }

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // ---------------------------------------------------------------------
    // Define input (X)
    // ---------------------------------------------------------------------
    auto X = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[0]);
    auto& X_node = builder.add_access(block, X);

    // ---------------------------------------------------------------------
    // Define weight (W)
    // ---------------------------------------------------------------------
    auto W = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[1]);
    auto& W_node = builder.add_access(block, W);

    // ---------------------------------------------------------------------
    // Optional bias (B)
    // ---------------------------------------------------------------------
    bool has_bias = (numOperands == 3);
    sdfg::data_flow::AccessNode* B_node_ptr = nullptr;
    if (has_bias) {
      auto B = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[2]);
      B_node_ptr = &builder.add_access(block, B);
    }

    // ---------------------------------------------------------------------
    // Define output (Y)
    // ---------------------------------------------------------------------
    auto Y = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto sdfg_type_Y = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(Y, *sdfg_type_Y);
    auto& Y_node = builder.add_access(block, Y);

    // ---------------------------------------------------------------------
    // Parse convolution attributes
    // ---------------------------------------------------------------------
    std::vector<size_t> dilations;
    std::vector<size_t> kernel_shape;
    std::vector<size_t> pads;
    std::vector<size_t> strides;

    for (auto namedAttr : libraryNodeOp->getAttrs()) {
      auto attrName = namedAttr.getName().getValue();
      if (attrName == "dilations") {
        dilations = attribute_to_size_t_vector(namedAttr.getValue());
      } else if (attrName == "kernel_shape") {
        kernel_shape = attribute_to_size_t_vector(namedAttr.getValue());
      } else if (attrName == "pads") {
        pads = attribute_to_size_t_vector(namedAttr.getValue());
      } else if (attrName == "strides") {
        strides = attribute_to_size_t_vector(namedAttr.getValue());
      }
    }

    // ---------------------------------------------------------------------
    // Create Conv library node
    // ---------------------------------------------------------------------
    auto& library_node = static_cast<sdfg::math::ml::ConvNode&>(
        builder.add_library_node<sdfg::math::ml::ConvNode>(
            block, sdfg::DebugInfo(), has_bias, dilations, kernel_shape, pads, strides));

    // ---------------------------------------------------------------------
    // Helper lambda for subset generation
    // ---------------------------------------------------------------------
    auto make_full_subset = [&](const std::string& container_name) {
      sdfg::data_flow::Subset begin_subset;
      sdfg::data_flow::Subset end_subset;
      const auto& t = sdfg.type(container_name);
      if (t.type_id() == sdfg::types::TypeID::Array) {
        sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(t), begin_subset, end_subset);
      } else {
        begin_subset.push_back(sdfg::symbolic::integer(0));
        end_subset.push_back(sdfg::symbolic::integer(0));
      }
      return std::make_pair(begin_subset, end_subset);
    };

    // ---------------------------------------------------------------------
    // Add input memlets (X, W, optional B)
    // ---------------------------------------------------------------------
    auto [begin_X, end_X] = make_full_subset(X);
    auto& iedge_X = builder.add_computational_memlet(block, X_node, library_node, "X", begin_X, end_X);

    auto [begin_W, end_W] = make_full_subset(W);
    auto& iedge_W = builder.add_computational_memlet(block, W_node, library_node, "W", begin_W, end_W);

    const sdfg::data_flow::Memlet* iedge_B_ptr = nullptr;
    sdfg::data_flow::Subset begin_B;
    sdfg::data_flow::Subset end_B;
    if (has_bias && B_node_ptr) {
      auto [b_begin, b_end] = make_full_subset(sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[2]));
      begin_B = b_begin;
      end_B = b_end;
      iedge_B_ptr = &builder.add_computational_memlet(block, *B_node_ptr, library_node, "B", begin_B, end_B);
    }

    // ---------------------------------------------------------------------
    // Add output memlet (Y)
    // ---------------------------------------------------------------------
    auto [begin_Y, end_Y] = make_full_subset(Y);
    auto& oedge_Y = builder.add_computational_memlet(block, library_node, "Y", Y_node, begin_Y, end_Y);

    // ---------------------------------------------------------------------
    // Expand library node
    // ---------------------------------------------------------------------
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    if (!library_node.expand(builder, analysis_manager)) {
      // Clean up on failure
      builder.remove_memlet(block, iedge_X);
      builder.remove_memlet(block, iedge_W);
      if (has_bias && iedge_B_ptr) {
        builder.remove_memlet(block, *iedge_B_ptr);
      }
      builder.remove_memlet(block, oedge_Y);
      builder.remove_node(block, library_node);
      builder.remove_node(block, X_node);
      builder.remove_node(block, W_node);
      if (has_bias && B_node_ptr) {
        builder.remove_node(block, *B_node_ptr);
      }
      builder.remove_node(block, Y_node);
      return false;
    }

    return true;
  }

  bool visit_relu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    if (libraryNodeOp.getOperands().size() != 1) {
      return false;
    }
    if (libraryNodeOp.getResults().size() != 1) {
      return false;
    }

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // Define input
    auto input = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[0]);
    auto& input_node = builder.add_access(block, input);

    // Define output
    auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto sdfg_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(output, *sdfg_type);
    auto& output_node = builder.add_access(block, output);

    auto& library_node = static_cast<sdfg::math::ml::ReLUNode&>(builder.add_library_node<sdfg::math::ml::ReLUNode>(block, sdfg::DebugInfo(), output, input));

    // Add input memlet
    auto& input_type = sdfg.type(input);
    sdfg::data_flow::Subset begin_subset;
    sdfg::data_flow::Subset end_subset;
    if (input_type.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset, end_subset);
    } else {
      begin_subset.push_back(sdfg::symbolic::integer(0));
      end_subset.push_back(sdfg::symbolic::integer(0));
    }
    auto& iedge = builder.add_computational_memlet(block, input_node, library_node, input, begin_subset, end_subset);

    // Add output memlet
    auto& oedge = builder.add_computational_memlet(block, library_node, output, output_node, begin_subset, end_subset);

    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    if (!library_node.expand(builder, analysis_manager)) {
      builder.remove_memlet(block, iedge);
      builder.remove_memlet(block, oedge);
      builder.remove_node(block, library_node);
      builder.remove_node(block, input_node);
      builder.remove_node(block, output_node);
      return false;
    }

    return true;
  }

  void visit_metadata_node(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // Add inputs
    std::vector<std::string> inputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> inputAccessNodes;
    for (auto arg : libraryNodeOp.getOperands()) {
      std::string input = sdfg::analysis::mlir_value_to_name(arg);
      inputs.push_back(input);
      auto& access_node = builder.add_access(block, input);
      inputAccessNodes[input] = &access_node;
    }

    // Add outputs
    std::vector<std::string> outputs;
    std::unordered_map<std::string, sdfg::data_flow::AccessNode*> outputAccessNodes;
    for (auto result : libraryNodeOp.getResults()) {
      std::string output = sdfg::analysis::mlir_value_to_name(result);
      auto sdfg_type = sdfg::analysis::mlir_type_to_sdfg_type(result.getType());
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
        sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset, end_subset);
      } else {
        begin_subset.push_back(sdfg::symbolic::integer(0));
        end_subset.push_back(sdfg::symbolic::integer(0));
      }
      builder.add_computational_memlet(block, *inputAccessNode, library_node, input, begin_subset, end_subset);
    }

    for (auto output : outputs) {
      auto outputAccessNode = outputAccessNodes[output];
      auto& output_type = sdfg.type(output);

      sdfg::data_flow::Subset begin_subset;
      sdfg::data_flow::Subset end_subset;
      if (output_type.type_id() == sdfg::types::TypeID::Array) {
        sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(output_type), begin_subset, end_subset);
      } else {
        begin_subset.push_back(sdfg::symbolic::integer(0));
        end_subset.push_back(sdfg::symbolic::integer(0));
      }
      builder.add_computational_memlet(block, library_node, output, *outputAccessNode, begin_subset, end_subset);
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
          std::string name = sdfg::analysis::mlir_value_to_name(sdfgNode.getArgument(argIdx++));
          auto sdfg_type = sdfg::analysis::mlir_type_to_sdfg_type(argType);
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