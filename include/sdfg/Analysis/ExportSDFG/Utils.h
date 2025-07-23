#pragma once

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>

#include <sdfg/builder/structured_sdfg_builder.h>

namespace sdfg {
namespace analysis {

inline std::unique_ptr<sdfg::types::IType> mlir_type_to_sdfg_type(mlir::Type type) {
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

  // Handle scalar types coming from the SDFG dialect itself
  if (auto scalarTy = mlir::dyn_cast<mlir::sdfg::ScalarType>(type)) {
    return mlir_type_to_sdfg_type(scalarTy.getPrimitiveType());
  }

  // Map the MLIR builtin none type to an SDFG void scalar.  This is used for
  // values that originate from `torch.constant.none`.
  if (mlir::isa<mlir::NoneType>(type)) {
    return std::make_unique<sdfg::types::Scalar>(
        sdfg::types::StorageType_CPU_Stack, 0, "",
        sdfg::types::PrimitiveType::Void);
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
inline std::string normalize_name(std::string name) {
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
inline std::string mlir_value_to_name(mlir::Value value) {
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

inline void sdfg_array_to_subset(const sdfg::types::Array& array, sdfg::data_flow::Subset& begin_subset, sdfg::data_flow::Subset& end_subset) {
  auto& element_type = array.element_type();
  auto num_elements = array.num_elements();

  begin_subset.push_back(sdfg::symbolic::integer(0));
  end_subset.push_back(num_elements);

  if (element_type.type_id() == sdfg::types::TypeID::Scalar) {
    return;
  }
  sdfg_array_to_subset(static_cast<const sdfg::types::Array&>(element_type), begin_subset, end_subset);
}

} // namespace analysis
} // namespace sdfg