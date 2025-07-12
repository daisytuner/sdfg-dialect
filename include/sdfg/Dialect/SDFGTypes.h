//===- SDFGTypes.h - SDFG dialect types -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SDFG_SDFGTYPES_H
#define SDFG_SDFGTYPES_H

#include "mlir/IR/BuiltinTypes.h"

// Generated classes for SDFG typedefs.
#define GET_TYPEDEF_CLASSES
#include "sdfg/Dialect/SDFGOpsTypes.h.inc"

#endif // SDFG_SDFGTYPES_H 