
# Enable perf
sudo sysctl kernel.perf_event_paranoid=0

# Run perf, support -d option (up to 3 times)
perf stat -d -d ./Multithreads

# perf can be used to record multithread app track for debug.

### PROFILING
# perf record ./binary
# perf report

# static is thread safe, think about overhead when use it. (some arch without atomic don't support it? https://youtu.be/JRUbzoVfkkw?t=4679)

Data race in single thread is possible.
Find an example

/*
Optimization abbreviations:
1. transposition TP
2. blocks        BL (arg - size of the block) https://en.wikipedia.org/wiki/Strassen_algorithm
3. multithreads  MT (arg - amount of threads)
4. vectorization VT (compile cond - support instructions) set automaticaly during compile time,
                     ability to use older SIMD/ARCH?
*/

Compiler flags:
-fopt-info-missed  show missing optimization

# Use cpupower to disable autoscaling
cpupower frequency-info -o proc
