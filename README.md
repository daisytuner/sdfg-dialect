# SDFG MLIR Dialect

An MLIR dialect based on the [C++ implementation of Stateful Dataflow Multigraphs](https://github.com/daisytuner/sdfglib).

## Requirements

### LLVM/MLIR

The project can be built out-of-tree with LLVM/MLIR versions installed via a package manager such as `apt`:

```bash
sudo apt-get install libmlir-19-dev mlir-19-tools
```
### LIT

The test suite uses LLVM's integrated testing tools `lit`, which can be installed from pip:

```bash
python3 -m venv venv
source venv/bin/activate

pip install lit
```

## Build

```bash
mkdir build && cd build
cmake -G Ninja -DMLIR_DIR=<path-to-MLIRConfig.cmake> -DLLVM_EXTERNAL_LIT=<path-to-lit> ..
ninja -j$(nproc)
```

### Running the Tests

```bash
cmake --build . --target check-sdfg-opt
```
