# ONNX Heap Analyzer

## Overview

ONNX Heap Analyzer is a tool designed to analyze memory usage patterns in ONNX models. It works by converting ONNX models to LLVM IR, building a control flow graph (CFG), extracting memory allocation/deallocation information, and simulating memory usage to predict peak memory requirements during model execution.

## Requirements

- Python 3.6+
- ONNX-MLIR toolchain
- LLVM toolchain (especially `opt` and `mlir-translate`)

## Usage

Basic usage:

```bash
python onnx_heap_analyzer.py path/to/model.onnx
```

This will perform the following steps:
1. Convert the ONNX model to LLVM IR
2. Apply optimization passes to simplify the IR
3. Extract the main graph output function
4. Parse basic blocks and build the control flow graph
5. Extract memory allocation and deallocation information
6. Simulate memory usage across all execution paths
7. Display peak memory usage and memory trace information

## Limitations

1. Currently only tracks simple `malloc`/`free` function calls (does not support `calloc`, `realloc`, etc.)
2. External tool dependencies (`onnx-mlir`, `mlir-translate`, `opt`) must be properly installed and available in PATH
