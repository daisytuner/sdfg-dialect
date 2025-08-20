#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_nodes/math/math.h>

#include "sdfg/Analysis/ExportSDFG/Utils.h"

namespace sdfg {
namespace analysis {
namespace nodes {

template<typename T>
bool visit_elementwise_unary(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
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
    auto& input_type = sdfg.type(input);
    auto& input_node = builder.add_access(block, input);

    // Define output
    auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto output_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(output, *output_type);
    auto& output_node = builder.add_access(block, output);

    auto& library_node = dynamic_cast<sdfg::math::ml::ElementWiseUnaryNode&>(builder.add_library_node<T>(block, sdfg::DebugInfo()));

    sdfg::data_flow::Subset begin_subset;
    sdfg::data_flow::Subset end_subset;
    if (output_type->type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(*output_type), begin_subset, end_subset);
    } else {
      begin_subset.push_back(sdfg::symbolic::integer(0));
      end_subset.push_back(sdfg::symbolic::integer(0));
    }

    auto& iedge = builder.add_computational_memlet(block, input_node, library_node, "X", begin_subset, end_subset, input_type);
    auto& oedge = builder.add_computational_memlet(block, library_node, "Y", output_node, begin_subset, end_subset, *output_type);

    return true;
}

template<typename T>
bool visit_elementwise_binary(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    if (libraryNodeOp.getOperands().size() != 2) {
      return false;
    }
    if (libraryNodeOp.getResults().size() != 1) {
      return false;
    }

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();
    auto& block = builder.add_block(root);

    // Define input
    auto input_a = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[0]);
    auto& input_type_a = sdfg.type(input_a);
    auto& input_node_a = builder.add_access(block, input_a);

    auto input_b = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[1]);
    auto& input_type_b = sdfg.type(input_b);
    auto& input_node_b = builder.add_access(block, input_b);

    // Define output
    auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto output_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(output, *output_type);
    auto& output_node = builder.add_access(block, output);

    auto& library_node = builder.add_library_node<T>(block, sdfg::DebugInfo());

    sdfg::data_flow::Subset begin_subset_out;
    sdfg::data_flow::Subset end_subset_out;
    if (output_type->type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(*output_type), begin_subset_out, end_subset_out);
    } else {
      begin_subset_out.push_back(sdfg::symbolic::integer(0));
      end_subset_out.push_back(sdfg::symbolic::integer(0));
    }

    sdfg::data_flow::Subset begin_subset_in_a;
    sdfg::data_flow::Subset end_subset_in_a;
    if (input_type_a.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type_a), begin_subset_in_a, end_subset_in_a);
    } else {
      begin_subset_in_a.push_back(sdfg::symbolic::integer(0));
      end_subset_in_a.push_back(sdfg::symbolic::integer(0));
    }

    sdfg::data_flow::Subset begin_subset_in_b;
    sdfg::data_flow::Subset end_subset_in_b;
    if (input_type_b.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type_b), begin_subset_in_b, end_subset_in_b);
    } else {
      begin_subset_in_b.push_back(sdfg::symbolic::integer(0));
      end_subset_in_b.push_back(sdfg::symbolic::integer(0));
    }

    // Add input memlet
    auto& iedge_a = builder.add_computational_memlet(block, input_node_a, library_node, "A", begin_subset_in_a, end_subset_in_a, input_type_a);
    auto& iedge_b = builder.add_computational_memlet(block, input_node_b, library_node, "B", begin_subset_in_b, end_subset_in_b, input_type_b);

    // Add output memlet
    auto& oedge = builder.add_computational_memlet(block, library_node, "C", output_node, begin_subset_out, end_subset_out, *output_type);

    return true;
}

template<typename T>
bool visit_reduce(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
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
  auto& input_type = sdfg.type(input);
  auto& input_node = builder.add_access(block, input);

  // Define output
  auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
  auto output_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
  builder.add_container(output, *output_type);
  auto& output_node = builder.add_access(block, output);

  // Define attributes
  int axis = -1;
  for (auto namedAttr : libraryNodeOp->getAttrs()) {
    auto attrName = namedAttr.getName().getValue();
    if (attrName == "axis") {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(namedAttr.getValue())) {
        axis = static_cast<int>(intAttr.getValue().getSExtValue());
      }
    }
  }

  auto& library_node = dynamic_cast<T&>(builder.add_library_node<T>(block, sdfg::DebugInfo(), axis));

  sdfg::data_flow::Subset begin_subset_out;
  sdfg::data_flow::Subset end_subset_out;
  if (output_type->type_id() == sdfg::types::TypeID::Array) {
    sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(*output_type), begin_subset_out, end_subset_out);
  } else {
    begin_subset_out.push_back(sdfg::symbolic::integer(0));
    end_subset_out.push_back(sdfg::symbolic::integer(0));
  }

  sdfg::data_flow::Subset begin_subset_in;
  sdfg::data_flow::Subset end_subset_in;
  if (input_type.type_id() == sdfg::types::TypeID::Array) {
    sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset_in, end_subset_in);
  } else {
    begin_subset_in.push_back(sdfg::symbolic::integer(0));
    end_subset_in.push_back(sdfg::symbolic::integer(0));
  }

  auto& iedge = builder.add_computational_memlet(block, input_node, library_node, "input", begin_subset_in, end_subset_in, input_type);
  auto& oedge = builder.add_computational_memlet(block, library_node, "output", output_node, begin_subset_out, end_subset_out, *output_type);

  return true;
}

bool visit_abs(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_add(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_clip(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_conv(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_div(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_dropout(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_elu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_erf(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_gemm(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_hard_sigmoid(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_leaky_relu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_log_softmax(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_matmul(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_maxpool(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_mul(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_pow(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_reduce_mean(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_relu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_sigmoid(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_softmax(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_sqrt(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_sub(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

bool visit_tanh(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp);

const std::unordered_map<std::string, bool (*)(sdfg::builder::StructuredSDFGBuilder&, mlir::sdfg::LibraryNodeOp)> LIBRARY_NODE_VISITORS = {
    {"ml::Abs", visit_abs},
    {"ml::Add", visit_add},
    {"ml::Clip", visit_clip},
    {"ml::Conv", visit_conv},
    {"ml::Div", visit_div},
    {"ml::Dropout", visit_dropout},
    {"ml::Elu", visit_elu},
    {"ml::Erf", visit_erf},
    {"ml::Gemm", visit_gemm},
    {"ml::HardSigmoid", visit_hard_sigmoid},
    {"ml::LeakyRelu", visit_leaky_relu},
    {"ml::LogSoftmax", visit_log_softmax},
    {"ml::MatMul", visit_matmul},
    {"ml::MaxPool", visit_maxpool},
    {"ml::Mul", visit_mul},
    {"ml::Pow", visit_pow},
    {"ml::ReduceMean", visit_reduce_mean},
    {"ml::Relu", visit_relu},
    {"ml::Sigmoid", visit_sigmoid},
    {"ml::Softmax", visit_softmax},
    {"ml::Sqrt", visit_sqrt},
    {"ml::Sub", visit_sub},
    {"ml::Tanh", visit_tanh},
};

} // namespace nodes
} // namespace analysis
} // namespace sdfg
