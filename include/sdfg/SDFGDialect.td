#ifndef SDFG_DIALECT
#define SDFG_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SDFG dialect definition.
//===----------------------------------------------------------------------===//

def SDFG_Dialect : Dialect {
    let name = "sdfg";
    let summary = "A SDFG MLIR dialect.";
    let description = [{
        This dialect is an MLIR dialect for SDFGs.
    }];
    let cppNamespace = "::mlir::sdfg";
}

//===----------------------------------------------------------------------===//
// Base SDFG operation definition.
//===----------------------------------------------------------------------===//

class SDFG_Op<string mnemonic, list<Trait> traits = []> :
        Op<SDFG_Dialect, mnemonic, traits>;

#endif // SDFG_DIALECT
