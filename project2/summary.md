# Benchmark Summary

## Environment

| Property  | Value                               |
| --------- | ----------------------------------- |
| CPU       | AMD Ryzen 5 3600                    |
| GPU       | NVIDIA GeForce RTX 3060             |
| Client OS | Linux 6.8.0-149-generic, glibc 2.39 |

## Benchmark configuration

| Parameter                | Value                                            |
| ------------------------ | ------------------------------------------------ |
| Fixed measurement window | 60 s                                             |
| Warm-up                  | At least 320 requests                            |
| Warm-up concurrency      | 16 clients                                       |
| Repeats                  | 3                                                |
| No batching              | Maximum batch size: 1; dynamic batching disabled |
| With batching            | Maximum batch size: 4; dynamic batching enabled  |

Latency percentiles are reported per completed HTTP request. Throughput is the
number of completed samples per 60-second measurement window. Values show the
mean ± standard deviation across three runs.

## Latency summary

| Concurrency | Configuration |     p50 (ms) |     p95 (ms) |     p99 (ms) |
| ----------: | ------------- | -----------: | -----------: | -----------: |
|           1 | No batching   |  3.88 ± 0.03 |  4.31 ± 0.04 |  4.70 ± 0.13 |
|          16 | No batching   | 43.48 ± 0.04 | 45.12 ± 0.08 | 46.10 ± 0.45 |
|          32 | No batching   | 88.65 ± 0.05 | 90.61 ± 0.31 | 97.84 ± 6.83 |
|           1 | With batching |  3.98 ± 0.09 |  4.37 ± 0.06 |  4.80 ± 0.11 |
|          16 | With batching | 42.78 ± 0.39 | 49.56 ± 6.79 | 53.81 ± 7.83 |
|          32 | With batching | 88.75 ± 0.02 | 92.06 ± 0.10 | 96.98 ± 6.16 |

## Benchmark summary

| Concurrency | Throughput — no batching (samples/s) | p95 latency — no batching (ms) | Average batch size — no batching | Throughput — with batching (samples/s) | p95 latency — with batching (ms) | Average batch size — with batching |
| ----------: | -----------------------------------: | -----------------------------: | -------------------------------: | -------------------------------------: | -------------------------------: | ---------------------------------: |
|           1 |                        193.34 ± 3.46 |                    4.31 ± 0.04 |                      1.00 ± 0.00 |                          186.49 ± 0.73 |                      4.37 ± 0.06 |                        1.00 ± 0.00 |
|          16 |                        353.63 ± 0.30 |                   45.12 ± 0.08 |                      1.00 ± 0.00 |                          341.39 ± 9.74 |                     49.56 ± 6.79 |                        4.00 ± 0.00 |
|          32 |                        353.03 ± 0.55 |                   90.61 ± 0.31 |                      1.00 ± 0.00 |                          346.56 ± 0.35 |                     92.06 ± 0.10 |                        4.00 ± 0.00 |
