#!/bin/bash

project_dir=$(realpath $(dirname $0))

mkdir -p "${project_dir}/build"
cd "${project_dir}/build"


conan install ../conanfile.txt --output-folder=. --build=missing


cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake  ..

#cmake -DCMAKE_TOOLCHAIN_FILE=./conan_paths.cmake  ..


