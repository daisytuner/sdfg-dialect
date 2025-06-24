// RUN: sdfg-opt %s | sdfg-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = sdfg.foo %{{.*}} : i32
        %res = sdfg.foo %0 : i32
        return
    }
}
