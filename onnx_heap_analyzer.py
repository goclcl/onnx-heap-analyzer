import subprocess
import os
import tempfile
import shutil
import sys
import argparse
import re
from collections import defaultdict


def convert_onnx_to_ll(input_path: str) -> str:
    """
    Converts an ONNX or MLIR file to LLVM IR (.ll) using:
      1. onnx-mlir --EmitLLVMIR -o <tmp/output> <input>  → <tmp/output>.onnx.mlir
      2. mlir-translate <tmp/output>.onnx.mlir -o <tmp/output>.ll
      3. deletes intermediate .onnx.mlir

    Returns:
        str: Path to the resulting .ll file
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = os.path.join(tmpdir, "output")
        mlir_path = f"{output_prefix}.onnx.mlir"
        ll_path = f"{output_prefix}.ll"

        # Step 1: Generate MLIR
        subprocess.run(
            ["onnx-mlir", "--EmitLLVMIR", "-o", output_prefix, input_path],
            check=True
        )
        if not os.path.exists(mlir_path):
            raise RuntimeError(f"onnx-mlir did not produce {mlir_path}")

        # Step 2: Translate MLIR to LLVM IR
        subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir", mlir_path, "-o", ll_path],
            check=True
        )
        if not os.path.exists(ll_path):
            raise RuntimeError(
                f"mlir-translate failed to generate .ll file: {ll_path}")

        # Step 3: Copy final .ll file to working dir
        final_ll_path = os.path.abspath("converted.ll")
        shutil.copyfile(ll_path, final_ll_path)

        # Done! tmpdir and .onnx.mlir are automatically deleted
        return final_ll_path


def run_instcombine_pass(ir_path: str) -> str:
    """
    Applies LLVM's `instcombine` pass to simplify pointer math and expressions.

    Args:
        ir_path (str): Path to the input LLVM IR file (.ll)

    Returns:
        str: Optimized IR as a string
    """
    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"IR file not found: {ir_path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ll") as tmp_out:
        optimized_path = tmp_out.name

    try:
        subprocess.run(
            ["opt", "-S", "-passes=instcombine", ir_path, "-o", optimized_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with open(optimized_path, "r") as f:
            simplified_ir = f.read()

    finally:
        os.remove(optimized_path)

    return simplified_ir


def extract_main_graph_output_body(ir_code: str) -> str:
    """
    Extracts the body of @main_graph_output from the given LLVM IR.

    Args:
        ir_code (str): Full LLVM IR code as a string.

    Returns:
        str: The body of @main_graph_output only (without define/closing brace).
    """
    in_func = False
    brace_level = 0
    func_lines = []

    for line in ir_code.splitlines():
        stripped = line.strip()

        # Start of the function
        if stripped.startswith("define") and "@main_graph_output" in stripped:
            in_func = True
            continue

        if in_func:
            if "{" in line:
                brace_level += 1
            if "}" in line:
                brace_level -= 1
                if brace_level <= 0:
                    break

            func_lines.append(line)

    if not func_lines:
        raise ValueError("Function @main_graph_output not found in IR.")

    return "\n".join(func_lines)


def parse_ir_into_blocks(ir_code: str) -> dict[int, list[str]]:
    """
    Parses LLVM IR code into basic blocks using numeric labels only.

    Args:
        ir_code (str): The complete LLVM IR code as a string.

    Returns:
        dict[int, list[str]]: Mapping from block number to a list of instructions.
    """
    blocks = defaultdict(list)
    current_block = 0  # Entry block (no label)

    for line in ir_code.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue  # Skip empty lines and full-line comments

        # Match labels like '42:', '42: ; preds = ...'
        match = re.match(r'^(\d+):\s*(;.*)?$', line)
        if match:
            current_block = int(match.group(1))
            continue

        blocks[current_block].append(line)

    return dict(blocks)


def build_cfg(blocks: dict[int, list[str]]) -> dict[int, list[int]]:
    """
    Builds a Control Flow Graph (CFG) from parsed LLVM IR blocks.

    Args:
        blocks (dict[int, list[str]]): Basic blocks from IR.

    Returns:
        dict[int, list[int]]: CFG mapping each block to its successor blocks.
    """
    cfg = defaultdict(list)

    for block_id, instrs in blocks.items():
        for instr in instrs:
            instr = instr.strip()

            # Match unconditional branch: br label %42
            match_uncond = re.match(r'br\s+label\s+%(\d+)', instr)
            if match_uncond:
                succ = int(match_uncond.group(1))
                cfg[block_id].append(succ)
                break

            # Match conditional branch: br i1 %cond, label %42, label %17
            match_cond = re.match(
                r'br\s+i1\s+%[\w\d]+,\s+label\s+%(\d+),\s+label\s+%(\d+)', instr)
            if match_cond:
                succ1 = int(match_cond.group(1))
                succ2 = int(match_cond.group(2))
                cfg[block_id].extend([succ1, succ2])
                break

        # Ensure all blocks appear in the CFG even if they have no successors
        cfg.setdefault(block_id, [])

    return dict(cfg)


def extract_malloc_free_info(blocks: dict[int, list[str]]):
    """
    Extracts malloc and free instructions from blocks.

    Returns:
        malloc_map (dict): {var_name: (block_id, size)}
        free_map   (dict): {var_name: block_id}
    """
    malloc_map = {}
    free_map = {}

    malloc_pattern = re.compile(
        r'^(%[\w\d]+)\s*=\s*call\s+ptr\s+@malloc\(\s*i64\s+(\d+)\s*\)')
    free_pattern = re.compile(
        r'^call\s+void\s+@free\(\s*ptr\s+(%[\w\d]+)\s*\)')

    for block_id, instrs in blocks.items():
        for instr in instrs:
            instr = instr.strip()

            m = malloc_pattern.match(instr)
            if m:
                var_name = m.group(1)
                size = int(m.group(2))
                malloc_map[var_name] = (block_id, size)
                continue

            f = free_pattern.match(instr)
            if f:
                var_name = f.group(1)
                free_map[var_name] = block_id

    return malloc_map, free_map


def simulate_memory_usage(cfg, blocks, malloc_map, free_map):
    """
    Simulates memory usage along all paths of the CFG using DFS.

    Args:
        cfg (dict[int, list[int]]): Control Flow Graph
        blocks (dict[int, list[str]]): Block instructions
        malloc_map (dict[str, (int, int)]): var → (block, size)
        free_map (dict[str, int]): var → block

    Returns:
        dict with:
            - peak_memory: int
            - memory_trace: list of (block, live_bytes)
    """
    visited = set()
    peak_memory = 0
    memory_trace = []

    def dfs(block_id, live_set):
        nonlocal peak_memory

        # Prevent re-entering same block in this path (to avoid infinite loops)
        state_id = (block_id, frozenset(live_set))
        if state_id in visited:
            return
        visited.add(state_id)

        # Copy live set for this path
        live_set = live_set.copy()

        # malloc: if variable allocated here
        for var, (blk, size) in malloc_map.items():
            if blk == block_id and var not in live_set:
                live_set[var] = size

        # free: if variable freed here
        for var, fblk in free_map.items():
            if fblk == block_id and var in live_set:
                del live_set[var]

        # Compute current live memory
        live_bytes = sum(live_set.values())
        memory_trace.append((block_id, live_bytes))
        peak_memory = max(peak_memory, live_bytes)

        # Recurse on successors
        for succ in cfg.get(block_id, []):
            dfs(succ, live_set)

    dfs(0, {})  # Start from block 0 with empty live set

    return {
        "peak_memory": peak_memory,
        "memory_trace": memory_trace
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ONNX model memory usage via LLVM IR.")
    parser.add_argument("onnx_path", type=str,
                        help="Path to input .onnx model")
    args = parser.parse_args()

    print("Step 0: Converting ONNX to LLVM IR...")
    ll_path = convert_onnx_to_ll(args.onnx_path)

    print("Step 1: Running instcombine optimization...")
    simplified_ir = run_instcombine_pass(ll_path)

    print("Step 2: Extracting @main_graph_output...")
    func_body = extract_main_graph_output_body(simplified_ir)

    print("Step 3: Parsing basic blocks...")
    blocks = parse_ir_into_blocks(func_body)
    print(f"    → Parsed {len(blocks)} blocks.")

    print("Step 4: Building CFG...")
    cfg = build_cfg(blocks)

    print("Step 5: Extracting malloc/free...")
    malloc_map, free_map = extract_malloc_free_info(blocks)
    print(f"    → Found {len(malloc_map)} mallocs and {len(free_map)} frees.")

    print("Step 6: Simulating memory usage...")
    result = simulate_memory_usage(cfg, blocks, malloc_map, free_map)

    print("\nAnalysis complete!")
    print(f"Peak memory usage: {result['peak_memory']} bytes")
    print("Memory trace (per block):")
    for blk, mem in result["memory_trace"]:
        print(f"  Block {blk}: {mem} bytes live")


if __name__ == "__main__":
    main()
