#include <gtest/gtest.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include "sdfg/Analysis/ExportSDFG/Passes.h"
#include "sdfg/Dialect/SDFGDialect.h"

#include <sdfg/structured_sdfg.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/data_flow/library_nodes/math/math.h>

#include <filesystem>

using namespace mlir;

class ReductionsTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    
    // Register the SDFG dialect properly
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::sdfg::SDFGDialect>();
    context->appendDialectRegistry(registry);
  }

  void TearDown() override {
    context.reset();
  }

  // Helper function to parse MLIR string and create a module
  OwningOpRef<ModuleOp> parseMLIR(const std::string& mlirStr) {
    return parseSourceString<ModuleOp>(mlirStr, context.get());
  }

  // Helper function to run the ExportSDFG pass
  LogicalResult runExportSDFGPass(ModuleOp module) {
    PassManager pm(context.get());
    pm.addPass(mlir::sdfg::analysis::createExportSDFGPass());
    return pm.run(module);
  }

  // Helper function to check if SDFG files were generated
  bool checkSDFGFilesGenerated(const std::string& sdfgName) {
    std::filesystem::path dotPath = sdfgName + ".dot";
    std::filesystem::path jsonPath = sdfgName + ".json";
    return std::filesystem::exists(dotPath) && std::filesystem::exists(jsonPath);
  }

  // Helper function to deserialize the SDFG file
  std::unique_ptr<::sdfg::StructuredSDFG> deserializeSDFGFile(const std::string& sdfgName) {
    std::filesystem::path jsonPath = sdfgName + ".json";
    std::ifstream file(jsonPath);

    json j;
    file >> j;
    ::sdfg::serializer::JSONSerializer serializer;
    return serializer.deserialize(j);
  }

  // Helper function to clean up generated files
  void cleanupGeneratedFiles(const std::string& sdfgName) {
      std::filesystem::path dotPath = sdfgName + ".dot";
      std::filesystem::path jsonPath = sdfgName + ".json";
      if (std::filesystem::exists(dotPath)) {
          std::filesystem::remove(dotPath);
      }
      if (std::filesystem::exists(jsonPath)) {
          std::filesystem::remove(jsonPath);
      }
  }

    std::unique_ptr<MLIRContext> context;
};

// Test Gemm operation
TEST_F(ReductionsTest, GemmOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @gemm_operation() {
  %0 = sdfg.alloca : !sdfg.array<32 x !sdfg.array<16 x f32>>
  %1 = sdfg.alloca : !sdfg.array<16 x !sdfg.array<64 x f32>>
  %2 = sdfg.library_node "ml::Gemm" %0, %1 : !sdfg.array<32 x !sdfg.array<16 x f32>>, !sdfg.array<16 x !sdfg.array<64 x f32>> -> !sdfg.array<32 x !sdfg.array<64 x f32>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.gemm_operation"));

  auto sdfg = deserializeSDFGFile("unknown_source.gemm_operation");
  EXPECT_EQ(sdfg->name(), "gemm_operation");
  EXPECT_EQ(sdfg->root().size(), 2);
  EXPECT_EQ(sdfg->containers().size(), 3);
  EXPECT_TRUE(sdfg->exists("_0"));
  EXPECT_TRUE(sdfg->exists("_1"));
  EXPECT_TRUE(sdfg->exists("_2"));

  auto& root = sdfg->root();
  auto block = dynamic_cast<::sdfg::structured_control_flow::Block*>(&root.at(0).first);
  EXPECT_NE(block, nullptr);

  auto& graph = block->dataflow();
  EXPECT_EQ(graph.nodes().size(), 4);
  EXPECT_EQ(graph.edges().size(), 3);

  bool found_lib_node = false;
  for (auto& node : graph.nodes()) {
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 2);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "ml::Gemm");

      auto& oedge = *graph.out_edges(*lib_node).begin();
      EXPECT_EQ(oedge.src_conn(), "Y");
      EXPECT_EQ(oedge.end_subset().size(), 2);
      EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));
      EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(1), ::sdfg::symbolic::integer(64)));

      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("unknown_source.gemm_operation");
}


// Test LogSoftmax operation
TEST_F(ReductionsTest, LogSoftmaxOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @log_softmax_operation() {
  %0 = sdfg.alloca : !sdfg.array<32 x !sdfg.array<16 x f32>>
  %1 = sdfg.library_node "ml::LogSoftmax" %0 : !sdfg.array<32 x !sdfg.array<16 x f32>> -> !sdfg.array<32 x f32> {axis = 1 : i64} 
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.log_softmax_operation"));

  auto sdfg = deserializeSDFGFile("unknown_source.log_softmax_operation");
  EXPECT_EQ(sdfg->name(), "log_softmax_operation");
  EXPECT_EQ(sdfg->root().size(), 2);
  EXPECT_EQ(sdfg->containers().size(), 2);
  EXPECT_TRUE(sdfg->exists("_0"));
  EXPECT_TRUE(sdfg->exists("_1"));

  auto& root = sdfg->root();
  auto block = dynamic_cast<::sdfg::structured_control_flow::Block*>(&root.at(0).first);
  EXPECT_NE(block, nullptr);

  auto& graph = block->dataflow();
  EXPECT_EQ(graph.nodes().size(), 3);
  EXPECT_EQ(graph.edges().size(), 2);

  bool found_lib_node = false;
  for (auto& node : graph.nodes()) {
    if (auto lib_node = dynamic_cast<::sdfg::math::ml::LogSoftmaxNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "ml::LogSoftmax");
      EXPECT_EQ(lib_node->axis(), 1);

      auto& oedge = *graph.out_edges(*lib_node).begin();
      EXPECT_EQ(oedge.src_conn(), "output");
      EXPECT_EQ(oedge.end_subset().size(), 1);
      EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));

      auto& iedge = *graph.in_edges(*lib_node).begin();
      EXPECT_EQ(iedge.dst_conn(), "input");
      EXPECT_EQ(iedge.begin_subset().size(), 2);
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(1), ::sdfg::symbolic::integer(16)));

      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("unknown_source.log_softmax_operation");
}

// Test MatMul operation
TEST_F(ReductionsTest, MatMulOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @matmul_operation() {
    %0 = sdfg.alloca : !sdfg.array<32 x !sdfg.array<16 x f32>>
    %1 = sdfg.alloca : !sdfg.array<16 x !sdfg.array<64 x f32>>
    %2 = sdfg.library_node "ml::MatMul" %0, %1 : !sdfg.array<32 x !sdfg.array<16 x f32>>, !sdfg.array<16 x !sdfg.array<64 x f32>> -> !sdfg.array<32 x !sdfg.array<64 x f32>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.matmul_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.matmul_operation");
    EXPECT_EQ(sdfg->name(), "matmul_operation");
    EXPECT_EQ(sdfg->root().size(), 2);
    EXPECT_EQ(sdfg->containers().size(), 3);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("_1"));
    EXPECT_TRUE(sdfg->exists("_2"));
  
    auto& root = sdfg->root();
    auto block = dynamic_cast<::sdfg::structured_control_flow::Block*>(&root.at(0).first);
    EXPECT_NE(block, nullptr);
  
    auto& graph = block->dataflow();
    EXPECT_EQ(graph.nodes().size(), 4);
    EXPECT_EQ(graph.edges().size(), 3);
  
    bool found_lib_node = false;
    for (auto& node : graph.nodes()) {
      if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
        EXPECT_FALSE(found_lib_node);
        EXPECT_EQ(graph.in_degree(*lib_node), 2);
        EXPECT_EQ(graph.out_degree(*lib_node), 1);
        EXPECT_EQ(lib_node->code().value(), "ml::MatMul");

        auto& oedge = *graph.out_edges(*lib_node).begin();
        EXPECT_EQ(oedge.src_conn(), "C");
        EXPECT_EQ(oedge.end_subset().size(), 2);
        EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));
        EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(1), ::sdfg::symbolic::integer(64)));

        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.matmul_operation");
}

// Test ReduceMean operation
TEST_F(ReductionsTest, ReduceMeanOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @reduce_mean_operation() {
  %0 = sdfg.alloca : !sdfg.array<32 x !sdfg.array<16 x f32>>
  %1 = sdfg.library_node "ml::ReduceMean" %0 : !sdfg.array<32 x !sdfg.array<16 x f32>> -> !sdfg.array<32 x f32> {axis = 1 : i64} 
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.reduce_mean_operation"));

  auto sdfg = deserializeSDFGFile("unknown_source.reduce_mean_operation");
  EXPECT_EQ(sdfg->name(), "reduce_mean_operation");
  EXPECT_EQ(sdfg->root().size(), 2);
  EXPECT_EQ(sdfg->containers().size(), 2);
  EXPECT_TRUE(sdfg->exists("_0"));
  EXPECT_TRUE(sdfg->exists("_1"));

  auto& root = sdfg->root();
  auto block = dynamic_cast<::sdfg::structured_control_flow::Block*>(&root.at(0).first);
  EXPECT_NE(block, nullptr);

  auto& graph = block->dataflow();
  EXPECT_EQ(graph.nodes().size(), 3);
  EXPECT_EQ(graph.edges().size(), 2);

  bool found_lib_node = false;
  for (auto& node : graph.nodes()) {
    if (auto lib_node = dynamic_cast<::sdfg::math::ml::ReduceMeanNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "ml::ReduceMean");
      EXPECT_EQ(lib_node->axis(), 1);

      auto& oedge = *graph.out_edges(*lib_node).begin();
      EXPECT_EQ(oedge.src_conn(), "output");
      EXPECT_EQ(oedge.end_subset().size(), 1);
      EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));

      auto& iedge = *graph.in_edges(*lib_node).begin();
      EXPECT_EQ(iedge.dst_conn(), "input");
      EXPECT_EQ(iedge.begin_subset().size(), 2);
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(1), ::sdfg::symbolic::integer(16)));

      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("unknown_source.reduce_mean_operation");
}

// Test Softmax operation
TEST_F(ReductionsTest, SoftmaxOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @softmax_operation() {
  %0 = sdfg.alloca : !sdfg.array<32 x !sdfg.array<16 x f32>>
  %1 = sdfg.library_node "ml::Softmax" %0 : !sdfg.array<32 x !sdfg.array<16 x f32>> -> !sdfg.array<32 x f32> {axis = 1 : i64} 
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.softmax_operation"));

  auto sdfg = deserializeSDFGFile("unknown_source.softmax_operation");
  EXPECT_EQ(sdfg->name(), "softmax_operation");
  EXPECT_EQ(sdfg->root().size(), 2);
  EXPECT_EQ(sdfg->containers().size(), 2);
  EXPECT_TRUE(sdfg->exists("_0"));
  EXPECT_TRUE(sdfg->exists("_1"));

  auto& root = sdfg->root();
  auto block = dynamic_cast<::sdfg::structured_control_flow::Block*>(&root.at(0).first);
  EXPECT_NE(block, nullptr);

  auto& graph = block->dataflow();
  EXPECT_EQ(graph.nodes().size(), 3);
  EXPECT_EQ(graph.edges().size(), 2);

  bool found_lib_node = false;
  for (auto& node : graph.nodes()) {
    if (auto lib_node = dynamic_cast<::sdfg::math::ml::SoftmaxNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "ml::Softmax");
      EXPECT_EQ(lib_node->axis(), 1);

      auto& oedge = *graph.out_edges(*lib_node).begin();
      EXPECT_EQ(oedge.src_conn(), "output");
      EXPECT_EQ(oedge.end_subset().size(), 1);
      EXPECT_TRUE(::sdfg::symbolic::eq(oedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));

      auto& iedge = *graph.in_edges(*lib_node).begin();
      EXPECT_EQ(iedge.dst_conn(), "input");
      EXPECT_EQ(iedge.begin_subset().size(), 2);
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(0), ::sdfg::symbolic::integer(32)));
      EXPECT_TRUE(::sdfg::symbolic::eq(iedge.end_subset().at(1), ::sdfg::symbolic::integer(16)));

      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("unknown_source.softmax_operation");
}

