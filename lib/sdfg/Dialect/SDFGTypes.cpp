//===- SDFGTypes.cpp - SDFG dialect types ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sdfg/Dialect/SDFGTypes.h"
#include "sdfg/Dialect/SDFGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::sdfg;

//===----------------------------------------------------------------------===//
// Generated type definitions.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "sdfg/Dialect/SDFGOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect registration of types.
//===----------------------------------------------------------------------===//

void SDFGDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "sdfg/Dialect/SDFGOpsTypes.cpp.inc"
      >();
} 