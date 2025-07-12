#ifndef SDFG_DIALECT
#define SDFG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"

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
    let useDefaultTypePrinterParser = 1;

    // Allow separate file to register custom types.
    let extraClassDeclaration = [{
        /// Register custom typedefs with this dialect.
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base SDFG operation definition.
//===----------------------------------------------------------------------===//

class SDFG_Op<string mnemonic, list<Trait> traits = []> :
        Op<SDFG_Dialect, mnemonic, traits>;

#endif // SDFG_DIALECT
