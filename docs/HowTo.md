
# Enable perf
`sudo sysctl kernel.perf_event_paranoid=0`

### PROFILING
Run `perf`, support -d option (up to 3 times)
```bash
perf stat -d -d {AppName}
perf record {AppName}
perf report
```

`perf` can be used to record multithread app track for debug.


### Optimization abbreviations
1. TP - transposition
2. BL - blocks(arg - size of the block) https://en.wikipedia.org/wiki/Strassen_algorithm
3. MT - multithreads(arg - amount of threads)
4. VT - vectorization(compile cond - support instructions) set automaticaly during compile time, ability to use older SIMD/ARCH?


Compiler flags:
-fopt-info-missed  show missing optimization

Use cpupower to disable autoscaling:
```bash
cpupower frequency-info -o proc
```

Get info about hw threads
```bash
lscpu -e
```

### Debugging

Dump core to the file for debuggind purpose
```bash
coredumpctl dump --output=core.dump
```

