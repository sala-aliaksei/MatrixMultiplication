#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sched.h>
#include <tracy/Tracy.hpp>
#include <vector>
#include <string>
#include <mutex>
#include <cstdio>


namespace tracy_utils
{

    enum class Metric {
        TotalCacheMisses,
        L1DataCacheMissRate,
        L1InstructionCacheMissRate,
        L2CacheMissRate,
        LLCCacheMissRate,
        BranchMisses,
        InstructionsRetired
    };

    class CacheMissTracer {
        int fd_miss = -1;
        int fd_access = -1;
        Metric metric;
        uint64_t last_miss_count = 0;
        uint64_t last_access_count = 0;

        static const char* GetPlotName(Metric m, int cpu) {
            static std::vector<std::string> storage;
            static std::vector<std::vector<const char*>> lookup; // [metric][cpu]
            static std::once_flag flag;

            std::call_once(flag, []() {
                const int MAX_CORES = 256;
                const int METRIC_COUNT = 7;
                const char* metricNames[] = {
                    "Total Cache Misses",
                    "L1 Data Cache Miss Rate",
                    "L1 Instr Cache Miss Rate",
                    "L2 Cache Miss Rate",
                    "LLC Cache Miss Rate",
                    "Branch Misses",
                    "Instructions Retired"
                };

                lookup.resize(METRIC_COUNT, std::vector<const char*>(MAX_CORES + 1));

                for (int m_idx = 0; m_idx < METRIC_COUNT; ++m_idx) {
                    for (int c = 0; c <= MAX_CORES; ++c) {
                        char buf[128];
                        if (c == MAX_CORES) {
                            snprintf(buf, sizeof(buf), "%s [Unknown Core]", metricNames[m_idx]);
                        } else {
                            snprintf(buf, sizeof(buf), "%s [Core %d]", metricNames[m_idx], c);
                        }
                        storage.push_back(buf);
                        lookup[m_idx][c] = storage.back().c_str();
                    }
                }
            });

            int m_idx = static_cast<int>(m);
            if (m_idx < 0 || m_idx >= 7) {
                m_idx = 0;
            }
            
            if (cpu < 0 || cpu >= 256) {
                cpu = 256; // Use the "Unknown Core" slot (last one)
            }

            return lookup[m_idx][cpu];
        }

    public:
        CacheMissTracer(Metric m = Metric::TotalCacheMisses) : metric(m) {
            struct perf_event_attr pe_miss = {};
            pe_miss.size = sizeof(struct perf_event_attr);
            pe_miss.disabled = 0;
            pe_miss.exclude_kernel = 1;
            pe_miss.exclude_hv = 1;

            struct perf_event_attr pe_access = {};
            pe_access.size = sizeof(struct perf_event_attr);
            pe_access.disabled = 0;
            pe_access.exclude_kernel = 1;
            pe_access.exclude_hv = 1;

            bool is_rate = false;

            switch (m) {
                case Metric::TotalCacheMisses:
                    pe_miss.type = PERF_TYPE_HARDWARE;
                    pe_miss.config = PERF_COUNT_HW_CACHE_MISSES;
                    break;
                case Metric::L1DataCacheMissRate:
                    is_rate = true;
                    pe_miss.type = PERF_TYPE_HW_CACHE;
                    pe_miss.config = (PERF_COUNT_HW_CACHE_L1D) | 
                                     (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                    pe_access.type = PERF_TYPE_HW_CACHE;
                    pe_access.config = (PERF_COUNT_HW_CACHE_L1D) | 
                                       (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                       (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                    break;
                case Metric::L1InstructionCacheMissRate:
                    is_rate = true;
                    pe_miss.type = PERF_TYPE_HW_CACHE;
                    pe_miss.config = (PERF_COUNT_HW_CACHE_L1I) | 
                                     (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                    pe_access.type = PERF_TYPE_HW_CACHE;
                    pe_access.config = (PERF_COUNT_HW_CACHE_L1I) | 
                                       (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                       (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                    break;
                case Metric::L2CacheMissRate:
                    is_rate = true;
                    pe_miss.type = PERF_TYPE_RAW;
                    pe_miss.config = 0x964; // l2_cache_req_stat.ic_dc_miss_in_l2
                    pe_access.type = PERF_TYPE_RAW;
                    pe_access.config = 0xf760; // l2_request_g1.all
                    break;
                case Metric::LLCCacheMissRate:
                    is_rate = true;
                    pe_miss.type = PERF_TYPE_HW_CACHE;
                    pe_miss.config = (PERF_COUNT_HW_CACHE_LL) | 
                                     (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                     (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
                    pe_access.type = PERF_TYPE_HW_CACHE;
                    pe_access.config = (PERF_COUNT_HW_CACHE_LL) | 
                                       (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
                                       (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
                    break;
                case Metric::BranchMisses:
                    pe_miss.type = PERF_TYPE_HARDWARE;
                    pe_miss.config = PERF_COUNT_HW_BRANCH_MISSES;
                    break;
                case Metric::InstructionsRetired:
                    pe_miss.type = PERF_TYPE_HARDWARE;
                    pe_miss.config = PERF_COUNT_HW_INSTRUCTIONS;
                    break;
                default:
                    pe_miss.type = PERF_TYPE_HARDWARE;
                    pe_miss.config = PERF_COUNT_HW_CACHE_MISSES;
                    break;
            }

            fd_miss = syscall(SYS_perf_event_open, &pe_miss, 0, -1, -1, 0);
            
            // Fallback for LLCCacheMissRate if specific event fails
            if (fd_miss == -1 && m == Metric::LLCCacheMissRate) {
                struct perf_event_attr fallback_pe = pe_miss;
                fallback_pe.type = PERF_TYPE_HARDWARE;
                fallback_pe.config = PERF_COUNT_HW_CACHE_MISSES;
                fd_miss = syscall(SYS_perf_event_open, &fallback_pe, 0, -1, -1, 0);
                
                // If fallback succeeds, try opening references as access
                if (fd_miss != -1) {
                    struct perf_event_attr fallback_access = pe_access;
                    fallback_access.type = PERF_TYPE_HARDWARE;
                    fallback_access.config = PERF_COUNT_HW_CACHE_REFERENCES;
                    fd_access = syscall(SYS_perf_event_open, &fallback_access, 0, -1, -1, 0);
                }
            } else if (is_rate && fd_miss != -1) {
                fd_access = syscall(SYS_perf_event_open, &pe_access, 0, -1, -1, 0);
            }

            if (fd_miss == -1) {
                // Determine the name of the failed metric for the error message
                const char* metricName = "Unknown";
                switch (m) {
                    case Metric::TotalCacheMisses: metricName = "TotalCacheMisses"; break;
                    case Metric::L1DataCacheMissRate: metricName = "L1DataCacheMissRate"; break;
                    case Metric::L1InstructionCacheMissRate: metricName = "L1InstructionCacheMissRate"; break;
                    case Metric::L2CacheMissRate: metricName = "L2CacheMissRate"; break;
                    case Metric::LLCCacheMissRate: metricName = "LLCCacheMissRate"; break;
                    case Metric::BranchMisses: metricName = "BranchMisses"; break;
                    case Metric::InstructionsRetired: metricName = "InstructionsRetired"; break;
                }
                char errBuf[128];
                snprintf(errBuf, sizeof(errBuf), "perf_event_open failed for %s", metricName);
                perror(errBuf);
            }
        }
    
        ~CacheMissTracer() { 
            if (fd_miss != -1) {
                close(fd_miss);
            }
            if (fd_access != -1) {
                close(fd_access);
            }
        }
    
        void update() {
            ZoneScoped;
            if (fd_miss == -1) {
                return;
            }

            uint64_t miss_count = 0;
            if (read(fd_miss, &miss_count, sizeof(uint64_t)) == sizeof(uint64_t)) {
                int cpu = sched_getcpu();
                int64_t delta_miss = static_cast<int64_t>(miss_count - last_miss_count);
                last_miss_count = miss_count;

                if (fd_access != -1) {
                    uint64_t access_count = 0;
                    if (read(fd_access, &access_count, sizeof(uint64_t)) == sizeof(uint64_t)) {
                        int64_t delta_access = static_cast<int64_t>(access_count - last_access_count);
                        last_access_count = access_count;

                        double rate = 0.0;
                        if (delta_access > 0) {
                            rate = static_cast<double>(delta_miss) / static_cast<double>(delta_access) * 100.0;
                        }
                        TracyPlot(GetPlotName(metric, cpu), rate);
                    }
                } else {
                    TracyPlot(GetPlotName(metric, cpu), delta_miss);
                }
            }
        }
    };

} // namespace tracy_utils
