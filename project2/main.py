import os
import platform
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import tritonclient.http as httpclient

MODEL_NAME_UNBATCHED = "resnet-classifier"
MODEL_NAME_BATCHED = "resnet-classifier-batched"
URL = "localhost:8000"
METRICS_URL = os.environ.get("TRITON_METRICS_URL", "http://localhost:8002/metrics")

CONCURRENT_CLIENTS = (1, 16, 32)
MEASURE_SECONDS = 60.0
WARMUP_REQUESTS = 320
REPEAT_COUNT = 3
WARMUP_CLIENTS = 16
BATCHING_PARAMETERS = "batched: max_batch_size=4, dynamic_batching=enabled; unbatched: max_batch_size=1, dynamic_batching=disabled"


def prepare_input() -> np.ndarray:
    # Random noise, we only care about the speed of the model, not the accuracy of the predictions.
    return np.random.rand(3, 224, 224).astype(np.float32).reshape(1, 3, 224, 224)


def make_request(
    client: httpclient.InferenceServerClient,
    input_data: np.ndarray,
    model_name: str,
) -> tuple[float, float]:
    inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [httpclient.InferRequestedOutput("output")]

    start = time.perf_counter()
    client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    end = time.perf_counter()
    return end - start, end


def warm_up(model_name: str, input_data: np.ndarray) -> None:
    def send_requests(_: int) -> None:
        client = httpclient.InferenceServerClient(url=URL)
        requests_per_client = WARMUP_REQUESTS // WARMUP_CLIENTS
        for _ in range(requests_per_client):
            make_request(client, input_data, model_name)

    print(f"Warming up {model_name}: at least {WARMUP_REQUESTS} requests with {WARMUP_CLIENTS} concurrent clients...")
    with ThreadPoolExecutor(max_workers=WARMUP_CLIENTS) as executor:
        list(executor.map(send_requests, range(WARMUP_CLIENTS)))


def get_triton_metric_counters(model_name: str) -> dict[str, float]:
    with urllib.request.urlopen(METRICS_URL, timeout=5) as response:
        metrics = response.read().decode("utf-8")

    counters = {
        "nv_inference_count": 0.0,
        "nv_inference_exec_count": 0.0,
    }
    found = set()

    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue

        name_and_labels, separator, value_text = line.partition(" ")
        if not separator:
            continue

        metric_name = name_and_labels.split("{", 1)[0]
        if metric_name not in counters:
            continue

        matches_model = True
        if "{" in name_and_labels:
            label_text = name_and_labels.split("{", 1)[1].rstrip("}")
            labels = {}
            for label in label_text.split(","):
                key, separator, value = label.strip().partition("=")
                if separator:
                    labels[key] = value.strip('"')
            matches_model = labels.get("model") == model_name

        if matches_model:
            try:
                counters[metric_name] += float(value_text)
                found.add(metric_name)
            except ValueError:
                continue

    missing = set(counters) - found
    if missing:
        raise RuntimeError(f"Triton metrics missing for {model_name}: {', '.join(sorted(missing))}")
    return counters


def client_loop(
    model_name: str,
    end_time: float,
) -> list[tuple[float, float]]:
    client = httpclient.InferenceServerClient(url=URL)
    latencies: list[tuple[float, float]] = []
    while time.perf_counter() < end_time:
        latency, completed_at = make_request(client, prepare_input(), model_name)
        if completed_at <= end_time:
            latencies.append((latency, completed_at))
    return latencies


def measure(
    model_name: str,
    concurrency: int,
) -> dict[str, float | int | str]:
    time.sleep(1)
    metrics_before = get_triton_metric_counters(model_name)
    end_time = time.perf_counter() + MEASURE_SECONDS
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(client_loop, model_name, end_time) for _ in range(concurrency)]
        request_results = [item for future in futures for item in future.result()]
    metrics_after = get_triton_metric_counters(model_name)

    completed_in_window = [latency for latency, completed_at in request_results if completed_at <= end_time]
    completed_count = len(completed_in_window)
    throughput = completed_count / MEASURE_SECONDS
    latencies_ms = np.asarray(completed_in_window, dtype=np.float64) * 1000.0
    inference_count_delta = metrics_after["nv_inference_count"] - metrics_before["nv_inference_count"]
    exec_count_delta = metrics_after["nv_inference_exec_count"] - metrics_before["nv_inference_exec_count"]
    average_batch_size = inference_count_delta / exec_count_delta if exec_count_delta > 0 else 0.0

    return {
        "model_name": model_name,
        "concurrency": concurrency,
        "completed_samples": completed_count,
        "throughput_samples_per_s": throughput,
        "triton_inference_count": inference_count_delta,
        "triton_exec_count": exec_count_delta,
        "average_batch_size": average_batch_size,
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "measurement_seconds": MEASURE_SECONDS,
    }


def hardware_description() -> str:
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unknown CPU"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = "unknown/CPU"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return f"CPU: {cpu}; GPU: {gpu}; client OS: {platform.platform()}"


def run_tests() -> pd.DataFrame:
    hardware = hardware_description()
    print(f"Hardware: {hardware}")
    print(f"Batching parameters: {BATCHING_PARAMETERS}")
    print(f"Fixed window: {MEASURE_SECONDS:.0f} s; warm-up: {WARMUP_REQUESTS} requests; repeats: {REPEAT_COUNT}")

    run_rows: list[dict[str, float | int | str]] = []
    for model_name in (MODEL_NAME_UNBATCHED, MODEL_NAME_BATCHED):
        warm_up(model_name, prepare_input())
        for concurrency in CONCURRENT_CLIENTS:
            for repeat in range(1, REPEAT_COUNT + 1):
                print(
                    f"Measuring {model_name} at concurrency {concurrency}: repeat {repeat}/{REPEAT_COUNT}",
                    flush=True,
                )
                row = measure(model_name, concurrency)
                row["repeat"] = repeat
                row["hardware"] = hardware
                row["batching_parameters"] = BATCHING_PARAMETERS
                row["payload_shape"] = str(prepare_input().shape)
                row["warmup_requests"] = WARMUP_REQUESTS
                run_rows.append(row)

    runs = pd.DataFrame(run_rows)
    runs.to_csv("benchmark_runs.csv", index=False)

    grouped = (
        runs.assign(
            batching=runs["model_name"].map(
                {
                    MODEL_NAME_UNBATCHED: "no batching",
                    MODEL_NAME_BATCHED: "with batching",
                }
            )
        )
        .groupby(["concurrency", "batching"], sort=False)
        .agg(
            throughput_mean=("throughput_samples_per_s", "mean"),
            throughput_std=("throughput_samples_per_s", "std"),
            batch_size_mean=("average_batch_size", "mean"),
            batch_size_std=("average_batch_size", "std"),
            p50_mean=("p50_ms", "mean"),
            p50_std=("p50_ms", "std"),
            p95_mean=("p95_ms", "mean"),
            p95_std=("p95_ms", "std"),
            p99_mean=("p99_ms", "mean"),
            p99_std=("p99_ms", "std"),
        )
        .reset_index()
    )

    latency_table = grouped.copy()
    for metric in ("p50", "p95", "p99"):
        latency_table[metric] = latency_table.apply(
            lambda row, metric=metric: f"{row[f'{metric}_mean']:.2f} +/- {row[f'{metric}_std']:.2f}",
            axis=1,
        )
    latency_table = latency_table[["concurrency", "batching", "p50", "p95", "p99"]].rename(
        columns={
            "concurrency": "Concurrency",
            "batching": "Configuration",
            "p50": "p50 [ms]",
            "p95": "p95 [ms]",
            "p99": "p99 [ms]",
        }
    )

    table = pd.DataFrame({"Concurrency": CONCURRENT_CLIENTS})
    for batching in ("no batching", "with batching"):
        subset = grouped[grouped["batching"] == batching].set_index("concurrency")
        table[f"Throughput {batching} [samples/s]"] = [
            f"{row['throughput_mean']:.2f} +/- {row['throughput_std']:.2f}" for row in (subset.loc[level] for level in CONCURRENT_CLIENTS)
        ]
        table[f"p95 latency {batching} [ms]"] = [
            f"{row['p95_mean']:.2f} +/- {row['p95_std']:.2f}" for row in (subset.loc[level] for level in CONCURRENT_CLIENTS)
        ]
        table[f"Avg batch size {batching}"] = [
            f"{row['batch_size_mean']:.2f} +/- {row['batch_size_std']:.2f}" for row in (subset.loc[level] for level in CONCURRENT_CLIENTS)
        ]

    print("\nSaved repeated run data to benchmark_runs.csv")
    print("Latency percentiles are per completed HTTP request (one sample/request); throughput = completed samples / 60 s.")
    print("\nLatency summary (mean +/- standard deviation across 3 runs):")
    print(latency_table.to_string(index=False))
    print("\nBenchmark summary:")
    print(table.to_string(index=False))
    return runs


if __name__ == "__main__":
    run_tests()
