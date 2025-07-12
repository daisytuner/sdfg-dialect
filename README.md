# SDFG MLIR Dialect

An MLIR dialect based on the [C++ implementation of Stateful Dataflow Multigraphs](https://github.com/daisytuner/sdfglib).

## Requirements

### LIT

The test suite uses LLVM's integrated testing tools `lit`, which can be installed from pip:

```bash
python3 -m venv venv
source venv/bin/activate

pip install lit
```

## Build

The following builds sdfg-dialect based on the IREE compiler packaging LLVM, MLIR and Torch-MLIR for us:

```bash
mkdir build && cd build

cmake -GNinja -DIREE_ENABLE_ASSERTIONS=ON -DIREE_ENABLE_SPLIT_DWARF=ON -DIREE_ENABLE_THIN_ARCHIVES=ON     -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_C_COMPILER_LAUNCHER=ccache     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON     -DIREE_HAL_DRIVER_DEFAULTS=OFF     -DIREE_HAL_DRIVER_LOCAL_SYNC=ON     -DIREE_HAL_DRIVER_LOCAL_TASK=ON     -DIREE_BUILD_PYTHON_BINDINGS=ON     -DPython3_EXECUTABLE="$(which python3)" -DCMAKE_BUILD_TYPE=Release -DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TOSA=OFF -DLLVM_EXTERNAL_LIT=<path-to-external-lit> ..

ninja -j$(nproc)
```

### Running the Tests

```bash
cmake --build . --target check-sdfg-opt
```
