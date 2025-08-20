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

#include <filesystem>

using namespace mlir;

class ElementwiseUnaryTest : public ::testing::Test {
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
      // if (std::filesystem::exists(dotPath)) {
      //     std::filesystem::remove(dotPath);
      // }
      if (std::filesystem::exists(jsonPath)) {
          std::filesystem::remove(jsonPath);
      }
  }

    std::unique_ptr<MLIRContext> context;
};

// Test Abs operation
TEST_F(ElementwiseUnaryTest, AbsOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @abs_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "Abs" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("abs_operation"));

  auto sdfg = deserializeSDFGFile("abs_operation");
  EXPECT_EQ(sdfg->name(), "abs_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "Abs");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("abs_operation");
}

// Test Clip operation
TEST_F(ElementwiseUnaryTest, ClipOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @clip_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "Clip" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("clip_operation"));

  auto sdfg = deserializeSDFGFile("clip_operation");
  EXPECT_EQ(sdfg->name(), "clip_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "Clip");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("clip_operation");
}

// Test Elu operation
TEST_F(ElementwiseUnaryTest, EluOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @elu_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "Elu" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("elu_operation"));

  auto sdfg = deserializeSDFGFile("elu_operation");
  EXPECT_EQ(sdfg->name(), "elu_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "Elu");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("elu_operation");
}

// Test Erf operation
TEST_F(ElementwiseUnaryTest, ErfOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @erf_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "Erf" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("erf_operation"));

  auto sdfg = deserializeSDFGFile("erf_operation");
  EXPECT_EQ(sdfg->name(), "erf_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "Erf");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("erf_operation");
}

// Test HardSigmoid operation
TEST_F(ElementwiseUnaryTest, HardSigmoidOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @hard_sigmoid_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "HardSigmoid" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("hard_sigmoid_operation"));

  auto sdfg = deserializeSDFGFile("hard_sigmoid_operation");
  EXPECT_EQ(sdfg->name(), "hard_sigmoid_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "HardSigmoid");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("hard_sigmoid_operation");
}

// Test LeakyReLU operation
TEST_F(ElementwiseUnaryTest, LeakyReluOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @leaky_relu_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "LeakyRelu" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("leaky_relu_operation"));

  auto sdfg = deserializeSDFGFile("leaky_relu_operation");
  EXPECT_EQ(sdfg->name(), "leaky_relu_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "LeakyReLU");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("leaky_relu_operation");
}

// Test ReLU operation
TEST_F(ElementwiseUnaryTest, ReluOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @relu_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.library_node "Relu" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("relu_operation"));

    auto sdfg = deserializeSDFGFile("relu_operation");
    EXPECT_EQ(sdfg->name(), "relu_operation");
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
      if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
        EXPECT_FALSE(found_lib_node);
        EXPECT_EQ(graph.in_degree(*lib_node), 1);
        EXPECT_EQ(graph.out_degree(*lib_node), 1);
        EXPECT_EQ(lib_node->code().value(), "ReLU");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("relu_operation");
}

// Test Sigmoid operation
TEST_F(ElementwiseUnaryTest, SigmoidOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @sigmoid_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.library_node "Sigmoid" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("sigmoid_operation"));

    auto sdfg = deserializeSDFGFile("sigmoid_operation");
    EXPECT_EQ(sdfg->name(), "sigmoid_operation");
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
      if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
        EXPECT_FALSE(found_lib_node);
        EXPECT_EQ(graph.in_degree(*lib_node), 1);
        EXPECT_EQ(graph.out_degree(*lib_node), 1);
        EXPECT_EQ(lib_node->code().value(), "Sigmoid");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("sigmoid_operation");
}

// Test Sqrt operation
TEST_F(ElementwiseUnaryTest, SqrtOperation) {
  const std::string mlirStr = R"(
module {
sdfg.sdfg @sqrt_operation() {
  %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  %1 = sdfg.library_node "Sqrt" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
  sdfg.return
}
}
)";

  auto module = parseMLIR(mlirStr);
  ASSERT_TRUE(module);
  
  auto result = runExportSDFGPass(*module);
  EXPECT_TRUE(succeeded(result));
  
  EXPECT_TRUE(checkSDFGFilesGenerated("sqrt_operation"));

  auto sdfg = deserializeSDFGFile("sqrt_operation");
  EXPECT_EQ(sdfg->name(), "sqrt_operation");
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
    if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
      EXPECT_FALSE(found_lib_node);
      EXPECT_EQ(graph.in_degree(*lib_node), 1);
      EXPECT_EQ(graph.out_degree(*lib_node), 1);
      EXPECT_EQ(lib_node->code().value(), "Sqrt");
      found_lib_node = true;
    }
  }
  EXPECT_TRUE(found_lib_node);

  cleanupGeneratedFiles("sqrt_operation");
}

// Test Tanh operation
TEST_F(ElementwiseUnaryTest, TanhOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @tanh_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.library_node "Tanh" %0 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("tanh_operation"));

    auto sdfg = deserializeSDFGFile("tanh_operation");
    EXPECT_EQ(sdfg->name(), "tanh_operation");
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
      if (auto lib_node = dynamic_cast<::sdfg::data_flow::LibraryNode*>(&node)) {
        EXPECT_FALSE(found_lib_node);
        EXPECT_EQ(graph.in_degree(*lib_node), 1);
        EXPECT_EQ(graph.out_degree(*lib_node), 1);
        EXPECT_EQ(lib_node->code().value(), "Tanh");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("tanh_operation");
}
