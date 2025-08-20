#include "sdfg/Analysis/ExportSDFG/PassDetail.h"
#include "sdfg/Analysis/ExportSDFG/Passes.h"

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGOps.h"
#include "sdfg/Analysis/ExportSDFG/Nodes.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#include <sdfg/data_flow/library_nodes/metadata_node.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/visualizer/dot_visualizer.h>

namespace {

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
    if (sdfg::analysis::nodes::LIBRARY_NODE_VISITORS.find(code) != sdfg::analysis::nodes::LIBRARY_NODE_VISITORS.end()) {
      success = sdfg::analysis::nodes::LIBRARY_NODE_VISITORS.at(code)(builder, libraryNodeOp);
    }

    if (!success) {
      visit_metadata_node(builder, libraryNodeOp);
    }
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
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(namedAttr.getValue())) {
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
      builder.add_computational_memlet(block, *inputAccessNode, library_node, input, begin_subset, end_subset, input_type);
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
      builder.add_computational_memlet(block, library_node, output, *outputAccessNode, begin_subset, end_subset, output_type);
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