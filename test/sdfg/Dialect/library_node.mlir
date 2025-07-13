// RUN: sdfg-opt %s | FileCheck %s

// CHECK: %0 = sdfg.library_node "add" %cst, %cst_0 : f64, f64 -> f64
sdfg.sdfg @library_node() -> f32 {
  %a = arith.constant 2.0 : f64
  %b = arith.constant 3.0 : f64
  %0 = sdfg.library_node "add" %a, %b : f64, f64 -> f64
  sdfg.return %0: f64
}
