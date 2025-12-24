# /// script
# dependencies = ["fastmcp"]
# ///
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LLVM-MCA")


def _select_compile_command(cc_path: Path, source_hint: Path | None) -> tuple[list[str], Path]:
    """Pick a compile command from compile_commands.json, optionally matching a source file."""
    entries = json.loads(cc_path.read_text())
    if not entries:
        raise RuntimeError("compile_commands.json is empty")

    def normalize(p: str) -> Path:
        return Path(p).expanduser().resolve()

    target = normalize(source_hint) if source_hint else None
    chosen = None
    for entry in entries:
        entry_path = normalize(entry["file"])
        if target and entry_path == target:
            chosen = entry
            break
    if chosen is None:
        chosen = entries[0]
        entry_path = normalize(chosen["file"])
    directory = Path(chosen.get("directory", ".")).expanduser()
    raw_cmd = chosen.get("arguments") or shlex.split(chosen["command"])
    return list(raw_cmd), entry_path, directory


def _run_clang_to_asm(
    tmpdir: Path,
    source_code: str,
    opt_level: str,
    std: str,
    compile_commands_path: Path | None,
    compile_source_hint: Path | None,
) -> Path:
    """Compile the provided C++ source code to an assembly file."""
    src_path = tmpdir / "snippet.cpp"
    asm_path = tmpdir / "snippet.s"
    src_path.write_text(source_code)

    if compile_commands_path and compile_commands_path.exists():
        base_cmd, original_src, workdir = _select_compile_command(
            compile_commands_path, compile_source_hint
        )
        compiler = base_cmd[0]
        args: list[str] = []
        skip_next = False
        for idx, arg in enumerate(base_cmd[1:]):
            if skip_next:
                skip_next = False
                continue
            if arg in {"-c", "-S", "-E"}:
                continue
            if arg == "-o":
                skip_next = True
                continue
            # Replace original source file with temp snippet if matched
            if Path(arg).resolve() == original_src:
                args.append(str(src_path))
                continue
            args.append(arg)
        # ensure source file present
        if all(Path(a) != src_path for a in map(Path, args)):
            args.append(str(src_path))
        clang_cmd = [compiler, "-S", "-o", str(asm_path), *args]
        subprocess.run(
            clang_cmd, check=True, capture_output=True, text=True, cwd=workdir
        )
    else:
        clang_cmd = [
            "clang++",
            f"-{opt_level}",
            f"-std={std}",
            "-S",
            "-o",
            str(asm_path),
            str(src_path),
        ]
        subprocess.run(clang_cmd, check=True, capture_output=True, text=True)
    return asm_path


@mcp.tool()
def analyze_source(
    source_code: str,
    cpu: str = "haswell",
    opt_level: str = "O3",
    std: str = "c++20",
    compile_commands_path: str | None = None,
    compile_source_hint: str | None = None,
) -> str:
    """
    Compiles C++ source to assembly with clang++ and analyzes it with llvm-mca.
    Returns throughput, resource pressure, and timeline views.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="llvm-mca-") as tmp:
            asm_path = _run_clang_to_asm(
                Path(tmp),
                source_code,
                opt_level,
                std,
                Path(compile_commands_path).expanduser()
                if compile_commands_path
                else None,
                Path(compile_source_hint).expanduser()
                if compile_source_hint
                else None,
            )
            process = subprocess.run(
                ["llvm-mca-20", f"-mcpu={cpu}", "-timeline", str(asm_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return process.stdout
    except subprocess.CalledProcessError as e:
        return (
            "Error running tool:\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        )
    except FileNotFoundError as e:
        missing = "clang++" if "clang" in str(e).lower() else "llvm-mca"
        return (
            f"Error: {missing} not found. "
            "Please install LLVM toolchain (e.g., sudo apt install llvm clang)."
        )


if __name__ == "__main__":
    mcp.run()