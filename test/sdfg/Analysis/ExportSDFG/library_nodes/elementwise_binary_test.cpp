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

class ElementwiseBinaryTest : public ::testing::Test {
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

// Test Add operation
TEST_F(ElementwiseBinaryTest, AddOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @add_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %2 = sdfg.library_node "ml::Add" %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.add_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.add_operation");
    EXPECT_EQ(sdfg->name(), "add_operation");
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
        EXPECT_EQ(lib_node->code().value(), "ml::Add");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.add_operation");
}

// Test Sub operation
TEST_F(ElementwiseBinaryTest, SubOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @sub_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %2 = sdfg.library_node "ml::Sub" %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.sub_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.sub_operation");
    EXPECT_EQ(sdfg->name(), "sub_operation");
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
        EXPECT_EQ(lib_node->code().value(), "ml::Sub");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.sub_operation");
}

// Test Mul operation
TEST_F(ElementwiseBinaryTest, MulOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @mul_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %2 = sdfg.library_node "ml::Mul" %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.mul_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.mul_operation");
    EXPECT_EQ(sdfg->name(), "mul_operation");
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
        EXPECT_EQ(lib_node->code().value(), "ml::Mul");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.mul_operation");
}

// Test Div operation
TEST_F(ElementwiseBinaryTest, DivOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @div_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %2 = sdfg.library_node "ml::Div" %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.div_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.div_operation");
    EXPECT_EQ(sdfg->name(), "div_operation");
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
        EXPECT_EQ(lib_node->code().value(), "ml::Div");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.div_operation");
}

// Test Pow operation
TEST_F(ElementwiseBinaryTest, PowOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @pow_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %1 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    %2 = sdfg.library_node "ml::Pow" %0, %1 : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>, !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>> -> !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("unknown_source.pow_operation"));

    auto sdfg = deserializeSDFGFile("unknown_source.pow_operation");
    EXPECT_EQ(sdfg->name(), "pow_operation");
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
        EXPECT_EQ(lib_node->code().value(), "ml::Pow");
        found_lib_node = true;
      }
    }
    EXPECT_TRUE(found_lib_node);

    cleanupGeneratedFiles("unknown_source.pow_operation");
}
