
deactivate_env() {
    unset ASAN_RT_DIR
    unset LD_LIBRARY_PATH
    unset LD_PRELOAD
}

activate_env() {
    export ASAN_RT_DIR="$(clang++$1 -print-resource-dir)/lib/linux"
    export LD_LIBRARY_PATH="$ASAN_RT_DIR:$LD_LIBRARY_PATH"
    export LD_PRELOAD="${ASAN_RT_DIR}/libclang_rt.asan-x86_64.so${LD_PRELOAD:+:$LD_PRELOAD}"
}

# If I source this script, I want to activate the environment
if [ "$0" != "$BASH_SOURCE" ]; then
    echo "Sourcing $BASH_SOURCE"
    if [ "$1" == "a" ] || [ "$1" == "activate" ]; then
        activate_env
        echo "ASAN environment activated"
    elif [ "$1" == "d" ] || [ "$1" == "deactivate" ]; then
        deactivate_env
        echo "ASAN environment deactivated"
    else
        echo "Usage: source $BASH_SOURCE <a,activate|d,deactivate>"
    fi
else
    echo "Not sourcing $BASH_SOURCE"
    echo "This script should be sourced, not executed directly"
    echo "Usage: source $BASH_SOURCE <a,activate|d,deactivate>"
    exit 1
fi