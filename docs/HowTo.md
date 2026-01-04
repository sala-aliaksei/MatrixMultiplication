
# Enable perf
`sudo sysctl kernel.perf_event_paranoid=0`

### Profiling
Run `perf`, support -d option (up to 3 times)
```bash
perf stat -d -d {AppName}
perf record {AppName}
perf report
```

`perf` can be used to record multithread app track for debug.

Compiler flags:
-fopt-info-missed  show missing optimization

Use cpupower to disable autoscaling:
```bash
cpupower frequency-info -o proc
```

Disable cpu scaling
```bash
sudo cpupower frequency-set --governor performance
```

Set high prioritet for process
```bash
sudo nice -n -20 ./app
```

Get info about hw threads
```bash
lscpu -e
```

 Disable ASLR
```cpp
    benchmark::MaybeReenterWithoutASLR(argc, argv);
```

### Debugging

Dump core to the file for debuggind purpose
```bash
coredumpctl dump --output=core.dump
```

