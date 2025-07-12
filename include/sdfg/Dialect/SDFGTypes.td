#ifndef SDFG_TYPES
#define SDFG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "sdfg/Dialect/SDFGDialect.td"

//===----------------------------------------------------------------------===//
// Base helpers
//===----------------------------------------------------------------------===//

class SDFG_Type<string name, list<Trait> traits = []> :
    TypeDef<SDFG_Dialect, name, traits>;

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

def SDFG_ArrayType : SDFG_Type<"Array"> {
  let mnemonic = "array";
  let summary = "Array with a single dimension and statically known size";
  let description = [{
    Represents a one-dimensional array with a compile-time constant length.
    The `size` field stores the number of elements as an unsigned 64-bit
    integer.  The contained *element type* may be any MLIR type, including
    another `sdfg.array`, allowing representation of higher-rank arrays via
    nesting.
  }];

  let parameters = (ins
    "uint64_t":$size,
    "::mlir::Type":$elementType
  );

  let skipDefaultBuilders = 0;

  let assemblyFormat = "`<` $size ` ` `x` ` ` $elementType `>`";
}

#endif // SDFG_TYPES