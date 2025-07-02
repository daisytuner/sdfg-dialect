// RUN: sdfg-opt %s | FileCheck %s

// CHECK: sdfg.symbol
// CHECK: sdfg.expression
sdfg.sdfg attributes {ID = 0 : i32, num_args = 0 : i32} {
  ^bb0:
    sdfg.symbol("N")
    %x = sdfg.expression("3*N+1") : i32
}
