import json, subprocess, os, shlex, sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ClangVerifier")

@mcp.tool()
def verify_file(file_path: str, build_dir: str = "build") -> str:
    """
    Verifies a single C++ file using the exact flags from compile_commands.json.
    Does a syntax-only check (-fsyntax-only) to avoid a full rebuild.
    """
    try:
        # 1. Find the compilation database
        db_path = os.path.join(os.getcwd(), build_dir, "compile_commands.json")
        if not os.path.exists(db_path):
            return "Error: compile_commands.json not found in /build"

        with open(db_path, 'r') as f:
            db = json.load(f)

        # 2. Match the file to its command
        abs_file_path = os.path.abspath(file_path)
        entry = next((e for e in db if os.path.abspath(os.path.join(e['directory'], e['file'])) == abs_file_path), None)

        if not entry:
            return f"Error: No entry found for {file_path} in compilation database."

        # 3. Modify the command for a fast syntax check
        # We replace the output file and add -fsyntax-only
        # compile_commands may store either "arguments" (array) or "command" (string)
        if 'arguments' in entry:
            args = list(entry['arguments'])
        elif 'command' in entry:
            args = shlex.split(entry['command'])
        else:
            return f"Error: No 'arguments' or 'command' field found for {file_path}."
        # Remove '-c' and the output filename if they exist to avoid creating objects
        if "-c" in args: args.remove("-c")
        if "-o" in args:
            idx = args.index("-o")
            args.pop(idx) # remove '-o'
            args.pop(idx) # remove the filename
        
        args.append("-fsyntax-only")

        print(f"DEBUG: Executing command: {' '.join(args)}", file=sys.stderr)
        # 4. Run the verification
        result = subprocess.run(
            args,
            cwd=entry['directory'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return f"✅ Verification successful for {file_path}"
        else:
            return f"❌ Verification failed:\n{result.stderr}"

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

if __name__ == "__main__":
    mcp.run()