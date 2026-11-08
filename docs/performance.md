# CPU profile

A local lexical indexing/search profile was run on 2026-11-08 using Python 3.12.12 on Windows 11, on an Intel 64-bit host (the runtime did not expose a friendlier CPU model string). The generated fixture contained 40 Python files and 1,000 small function definitions. This is a reproducible smoke workload, not a representative external codebase.

| Measurement | Result |
|---|---:|
| Indexing wall time | 0.814 s |
| Indexed functions | 1,000 |
| SQLite database size | 770,048 bytes |
| Lexical queries | 100 |
| Mean query latency | 3.754 ms |
| p95 query latency | 6.505 ms |
| Process RSS at end | 28.7 MiB |
| Python traced peak during run | 1.1 MiB |

The profile used identical query text for the 100 repeated FTS5 lexical searches. RSS is whole-process memory at the end of the run; the Python allocation peak is `tracemalloc` only and excludes native allocations. No semantic model or FAISS vectors were loaded. These measurements are host-specific and should not be used to predict performance on other repositories or machines.

The exact FAISS index remains the correctness baseline. No approximate index or recall-affecting tuning was introduced based on this small profile.
