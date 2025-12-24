import subprocess
from mcp.server.fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("ClangTidyServer")

@mcp.tool()
def run_clang_tidy(file_path: str) -> str:
    """
    Runs clang-tidy on a specific C++ file and returns the warnings/errors.
    Make sure a compile_commands.json exists in your build directory.
    """
    try:
        # Command to run clang-tidy. 
        # '-p build' tells it to look for compile_commands.json in the /build folder.
        result = subprocess.run(
            ["clang-tidy", file_path, "-p", "build", "--quiet"],
            capture_output=True,
            text=True
        )
        
        if not result.stdout and not result.stderr:
            return f"✅ No issues found in {file_path}"
        
        return result.stdout + result.stderr
    except Exception as e:
        return f"❌ Error running clang-tidy: {str(e)}"

if __name__ == "__main__":
    mcp.run()