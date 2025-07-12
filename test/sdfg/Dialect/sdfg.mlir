// RUN: sdfg-opt %s | FileCheck %s

// CHECK: sdfg.sdfg @empty() -> f32
sdfg.sdfg @empty() -> f32 {
  sdfg.return
}

// CHECK: sdfg.sdfg @with_args(%arg0: f32) -> f32
sdfg.sdfg @with_args(%arg0: f32) -> f32 {
  sdfg.return %arg0 : f32
}
