#include "sdfg/Analysis/ExportSDFG/Nodes.h"

namespace sdfg {
namespace analysis {
namespace nodes {

std::vector<size_t> attribute_to_size_t_vector(mlir::Attribute attr) {
    std::vector<size_t> result;
    
    // Case 1: ArrayAttr of IntegerAttr
    if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
        result.reserve(arr.size());
        for (auto element : arr) {
        if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element)) {
            // Use APInt helpers to avoid sign-ness assertion in getInt().
            result.push_back(static_cast<size_t>(intAttr.getValue().getZExtValue()));
        }
        }
        return result;
    }
    
    // Case 2: Dense int elements (e.g., DenseI64ArrayAttr, DenseElementsAttr of ints)
    if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
        result.reserve(static_cast<size_t>(dense.size()));
        for (auto value : dense.getValues<llvm::APInt>()) {
        result.push_back(static_cast<size_t>(value.getZExtValue()));
        }
        return result;
    }
    
    // Fallback: return empty vector if unsupported type.
    return result;
}

bool visit_abs(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::AbsNode>(builder, libraryNodeOp);
}

bool visit_add(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::AddNode>(builder, libraryNodeOp);
}

bool visit_clip(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::ClipNode>(builder, libraryNodeOp);
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
    auto& sdfg_type_X = builder.subject().type(X);
    auto& X_node = builder.add_access(block, X);

    // ---------------------------------------------------------------------
    // Define weight (W)
    // ---------------------------------------------------------------------
    auto W = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[1]);
    auto& sdfg_type_W = builder.subject().type(W);
    auto& W_node = builder.add_access(block, W);

    // ---------------------------------------------------------------------
    // Optional bias (B)
    // ---------------------------------------------------------------------
    bool has_bias = (numOperands == 3);
    sdfg::data_flow::AccessNode* B_node_ptr = nullptr;
    const sdfg::types::IType* sdfg_type_B = nullptr;
    if (has_bias) {
        auto B = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[2]);
        B_node_ptr = &builder.add_access(block, B);
        sdfg_type_B = &builder.subject().type(B);
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
    auto& iedge_X = builder.add_computational_memlet(block, X_node, library_node, "X", begin_X, end_X, sdfg_type_X);

    auto [begin_W, end_W] = make_full_subset(W);
    auto& iedge_W = builder.add_computational_memlet(block, W_node, library_node, "W", begin_W, end_W, sdfg_type_W);

    const sdfg::data_flow::Memlet* iedge_B_ptr = nullptr;
    sdfg::data_flow::Subset begin_B;
    sdfg::data_flow::Subset end_B;
    if (has_bias && B_node_ptr) {
        auto [b_begin, b_end] = make_full_subset(sdfg::analysis::mlir_value_to_name(libraryNodeOp.getOperands()[2]));
        begin_B = b_begin;
        end_B = b_end;
        iedge_B_ptr = &builder.add_computational_memlet(block, *B_node_ptr, library_node, "B", begin_B, end_B, *sdfg_type_B);
    }

    // ---------------------------------------------------------------------
    // Add output memlet (Y)
    // ---------------------------------------------------------------------
    auto [begin_Y, end_Y] = make_full_subset(Y);
    auto& oedge_Y = builder.add_computational_memlet(block, library_node, "Y", Y_node, begin_Y, end_Y, *sdfg_type_Y);

    return true;
}

bool visit_div(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::DivNode>(builder, libraryNodeOp);
}

bool visit_dropout(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::DropoutNode>(builder, libraryNodeOp);
}

bool visit_elu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::EluNode>(builder, libraryNodeOp);
}

bool visit_erf(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::ErfNode>(builder, libraryNodeOp);
}

bool visit_hard_sigmoid(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::HardSigmoidNode>(builder, libraryNodeOp);
}

bool visit_leaky_relu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::LeakyReLUNode>(builder, libraryNodeOp);
}

bool visit_matmul(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::MatMulNode>(builder, libraryNodeOp);
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
    auto& input_type = builder.subject().type(input);
    auto& input_node = builder.add_access(block, input);

    // Define output
    auto output = sdfg::analysis::mlir_value_to_name(libraryNodeOp.getResults()[0]);
    auto output_type = sdfg::analysis::mlir_type_to_sdfg_type(libraryNodeOp.getResults()[0].getType());
    builder.add_container(output, *output_type);
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

    sdfg::data_flow::Subset begin_subset_in;
    sdfg::data_flow::Subset end_subset_in;
    if (input_type.type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(input_type), begin_subset_in, end_subset_in);
    } else {
      begin_subset_in.push_back(sdfg::symbolic::integer(0));
      end_subset_in.push_back(sdfg::symbolic::integer(0));
    }
    auto& iedge = builder.add_computational_memlet(block, input_node, library_node, "X", begin_subset_in, end_subset_in, input_type);
  
    sdfg::data_flow::Subset begin_subset_out;
    sdfg::data_flow::Subset end_subset_out;
    if (output_type->type_id() == sdfg::types::TypeID::Array) {
      sdfg::analysis::sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(*output_type), begin_subset_out, end_subset_out);
    } else {
      begin_subset_out.push_back(sdfg::symbolic::integer(0));
      end_subset_out.push_back(sdfg::symbolic::integer(0));
    }
    auto& oedge = builder.add_computational_memlet(block, library_node, "Y", output_node, begin_subset_out, end_subset_out, *output_type);

    return true;
}

bool visit_mul(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::MulNode>(builder, libraryNodeOp);
}

bool visit_pow(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::PowNode>(builder, libraryNodeOp);
}

bool visit_relu(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::ReLUNode>(builder, libraryNodeOp);
}

bool visit_sigmoid(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::SigmoidNode>(builder, libraryNodeOp);
}

bool visit_sqrt(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::SqrtNode>(builder, libraryNodeOp);
}

bool visit_sub(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_binary<sdfg::math::ml::SubNode>(builder, libraryNodeOp);
}

bool visit_tanh(sdfg::builder::StructuredSDFGBuilder& builder, mlir::sdfg::LibraryNodeOp libraryNodeOp) {
    return visit_elementwise_unary<sdfg::math::ml::TanhNode>(builder, libraryNodeOp);
}

} // namespace nodes
} // namespace analysis
} // namespace sdfg