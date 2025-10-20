# conanfile.py — MatrixMultiplication (Conan v2)
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
from conan.tools.build import can_run

class MatrixMultiplicationRecipe(ConanFile):
    name = "MatrixMultiplication"
    version = "0.1.0"
    package_type = "library"
    license = "Apache-2.0"
    author = "Your Name <you@example.com>"
    url = "https://example.com/MatrixMultiplication"
    description = "High-performance matrix multiplication benchmarks and tests."
    settings = ("os", "arch", "compiler", "build_type")

    options = {
        "shared": [True, False],
    }
    default_options = {
        "shared": True,
    }

    # If you plan to 'conan create', include your sources here.
    # Keep paths minimal and adjust to your tree.
    export_sources = (
        "CMakeLists.txt",
        "src/*",
    )

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("benchmark/1.9.4")
        self.requires("gtest/1.17.0")
        #self.requires("boost/1.83.0")
        self.requires("openblas/0.3.30")
        self.requires("eigen/5.0.0")

        # self.requires("tensorflow-lite/2.12.0")  # Uncomment if/when needed
        #self.requires("llvm-openmp/20.1.6")
                

    def build_requirements(self):
        # Pin reasonable versions; feel free to loosen/pin as your env needs
        self.tool_requires("cmake/[>3.29.6]")
        self.tool_requires("ninja/[>1.11.0]")


    def configure(self):
        # Trim Boost if you don’t need heavy components (optional)
        # self.options["boost"].without_python = True
        # self.options["boost"].without_graph = True
        # self.options["boost"].without_graph_parallel = True
        # self.options["boost"].without_mpi = True
        # self.options["boost"].without_stacktrace = True
        # self.options["boost"].without_test = True

        self.options["openblas"].shared = True
        self.options["eigen"].shared = True

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        # Prefer Ninja for speed; remove if you want default generator
        tc.generator = "Ninja"

        # Example: set C++ standard via toolchain (can also be done in profile)
        # tc.variables["CMAKE_CXX_STANDARD"] = "20"

        # On some platforms you may want to force find_package(OpenMP)
        # through flags, but llvm-openmp package typically handles it.
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()


    def package(self):
        pass

    def package_info(self):
        pass


