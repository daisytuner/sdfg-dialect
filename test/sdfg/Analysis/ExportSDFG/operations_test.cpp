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
#include "sdfg/Dialect/SDFGOpsDialect.cpp.inc"

#include <sdfg/structured_sdfg.h>
#include <sdfg/serializer/json_serializer.h>

#include <filesystem>

using namespace mlir;
using json = nlohmann::json;

class OperationsTest : public ::testing::Test {
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

// Test return operation
TEST_F(OperationsTest, ReturnOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @return_operation() {
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    // Check if files were generated
    EXPECT_TRUE(checkSDFGFilesGenerated("return_operation"));
    
    auto sdfg = deserializeSDFGFile("return_operation");
    EXPECT_EQ(sdfg->name(), "return_operation");

    EXPECT_EQ(sdfg->root().size(), 1);
    auto return_node = dynamic_cast<::sdfg::structured_control_flow::Return*>(&sdfg->root().at(0).first);
    EXPECT_NE(return_node, nullptr);

    // Clean up
    cleanupGeneratedFiles("return_operation");
}

// Test alloca operation
TEST_F(OperationsTest, AllocaOperation) {
    const std::string mlirStr = R"(
module {
  sdfg.sdfg @alloca_operation() {
    %0 = sdfg.alloca : !sdfg.array<1 x !sdfg.array<3 x !sdfg.array<224 x !sdfg.array<224 x f32>>>>
    sdfg.return
  }
}
)";

    auto module = parseMLIR(mlirStr);
    ASSERT_TRUE(module);
    
    auto result = runExportSDFGPass(*module);
    EXPECT_TRUE(succeeded(result));
    
    EXPECT_TRUE(checkSDFGFilesGenerated("alloca_operation"));

    auto sdfg = deserializeSDFGFile("alloca_operation");
    EXPECT_EQ(sdfg->name(), "alloca_operation");

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("_0"));
    auto& type = sdfg->type("_0");
    EXPECT_EQ(type.type_id(), ::sdfg::types::TypeID::Array);

    EXPECT_EQ(sdfg->root().size(), 1);
    auto return_node = dynamic_cast<::sdfg::structured_control_flow::Return*>(&sdfg->root().at(0).first);
    EXPECT_NE(return_node, nullptr);

    cleanupGeneratedFiles("alloca_operation");
}
