from __future__ import annotations

import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_LENGTH = 128
WARMUP_RUNS = 20
INDIVIDUAL_RUNS = 300
STABILITY_RUNS = 500
BATCH_SIZES = [1, 4, 8, 16, 32]
BATCH_REPETITIONS = 20
CONFIDENCE_TOLERANCE = 1e-4

LABELS = [
    "Contratos",
    "Facturas",
    "Colaboraciones",
    "Clientes",
    "Publicidad",
    "Varios",
]

ID_TO_LABEL = {
    index: label for index, label in enumerate(LABELS)
}


def bytes_to_mib(value: int | float) -> float:
    return round(float(value) / (1024**2), 2)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


class ResourceMonitor:
    """Muestrea uso de CPU, RAM del proceso y memoria CUDA durante el benchmark."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.records: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        self.process.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            timestamp = time.perf_counter()
            cpu_percent = self.process.cpu_percent(interval=None)
            rss_mib = bytes_to_mib(self.process.memory_info().rss)
            system_ram_percent = psutil.virtual_memory().percent

            gpu_allocated_mib = 0.0
            gpu_reserved_mib = 0.0

            if torch.cuda.is_available():
                gpu_allocated_mib = bytes_to_mib(
                    torch.cuda.memory_allocated()
                )
                gpu_reserved_mib = bytes_to_mib(
                    torch.cuda.memory_reserved()
                )

            self.records.append(
                {
                    "timestamp_perf_counter": timestamp,
                    "process_cpu_percent": float(cpu_percent),
                    "process_rss_mib": float(rss_mib),
                    "system_ram_percent": float(system_ram_percent),
                    "gpu_allocated_mib": float(gpu_allocated_mib),
                    "gpu_reserved_mib": float(gpu_reserved_mib),
                }
            )

            self._stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)


def prepare_batch(
    texts: list[str],
    tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    return {
        key: value.to(device, non_blocking=True)
        for key, value in encoded.items()
    }


def infer_batch(
    encoded: dict[str, torch.Tensor],
    model,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            outputs = model(**encoded)

    logits = outputs.logits.detach().float()
    probabilities = torch.softmax(logits, dim=-1)
    confidences, predicted_ids = probabilities.max(dim=-1)

    return predicted_ids.cpu(), confidences.cpu()


def benchmark_individual_latency(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_index in range(INDIVIDUAL_RUNS):
        text = texts[run_index % len(texts)]
        encoded = prepare_batch([text], tokenizer, device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        predicted_ids, confidences = infer_batch(
            encoded,
            model,
        )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        rows.append(
            {
                "run": run_index + 1,
                "latency_ms": elapsed_ms,
                "predicted_id": int(predicted_ids[0]),
                "predicted_label": ID_TO_LABEL[
                    int(predicted_ids[0])
                ],
                "confidence": float(confidences[0]),
            }
        )

    return pd.DataFrame(rows)


def benchmark_throughput(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for batch_size in BATCH_SIZES:
        selected_texts = [
            texts[index % len(texts)]
            for index in range(batch_size)
        ]

        encoded = prepare_batch(
            selected_texts,
            tokenizer,
            device,
        )

        _ = infer_batch(encoded, model)
        torch.cuda.synchronize()

        elapsed_times: list[float] = []

        for _ in range(BATCH_REPETITIONS):
            torch.cuda.synchronize()
            start = time.perf_counter()

            predicted_ids, confidences = infer_batch(
                encoded,
                model,
            )

            torch.cuda.synchronize()
            elapsed_seconds = time.perf_counter() - start
            elapsed_times.append(elapsed_seconds)

            if not torch.isfinite(confidences).all().item():
                raise RuntimeError(
                    f"Se detectaron confidencias no finitas con batch={batch_size}."
                )

        mean_seconds = statistics.mean(elapsed_times)
        p95_seconds = percentile(elapsed_times, 95)

        rows.append(
            {
                "batch_size": batch_size,
                "repetitions": BATCH_REPETITIONS,
                "mean_batch_latency_ms": mean_seconds * 1000,
                "p95_batch_latency_ms": p95_seconds * 1000,
                "mean_latency_per_email_ms": (
                    mean_seconds / batch_size * 1000
                ),
                "throughput_emails_per_second": (
                    batch_size / mean_seconds
                ),
            }
        )

    return pd.DataFrame(rows)


def run_stability_test(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> dict[str, Any]:
    reference: dict[int, tuple[int, float]] = {}
    inconsistencies = 0
    non_finite_outputs = 0
    exceptions = 0
    latencies_ms: list[float] = []

    for run_index in range(STABILITY_RUNS):
        text_index = run_index % len(texts)
        text = texts[text_index]

        try:
            encoded = prepare_batch(
                [text],
                tokenizer,
                device,
            )

            torch.cuda.synchronize()
            start = time.perf_counter()

            predicted_ids, confidences = infer_batch(
                encoded,
                model,
            )

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

            predicted_id = int(predicted_ids[0])
            confidence = float(confidences[0])

            if not math.isfinite(confidence):
                non_finite_outputs += 1
                continue

            if text_index not in reference:
                reference[text_index] = (
                    predicted_id,
                    confidence,
                )
            else:
                reference_id, reference_confidence = reference[
                    text_index
                ]

                same_label = predicted_id == reference_id
                confidence_stable = (
                    abs(confidence - reference_confidence)
                    <= CONFIDENCE_TOLERANCE
                )

                if not (same_label and confidence_stable):
                    inconsistencies += 1

        except Exception:
            exceptions += 1

    success_count = (
        STABILITY_RUNS
        - inconsistencies
        - non_finite_outputs
        - exceptions
    )

    return {
        "runs": STABILITY_RUNS,
        "success_count": success_count,
        "success_rate": success_count / STABILITY_RUNS,
        "inconsistencies": inconsistencies,
        "non_finite_outputs": non_finite_outputs,
        "exceptions": exceptions,
        "confidence_tolerance": CONFIDENCE_TOLERANCE,
        "latency_mean_ms": statistics.mean(latencies_ms),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_max_ms": max(latencies_ms),
    }


def summarize_resources(
    records: list[dict[str, float]],
) -> dict[str, Any]:
    if not records:
        return {
            "samples": 0,
            "process_cpu_mean_percent": None,
            "process_cpu_max_percent": None,
            "process_rss_mean_mib": None,
            "process_rss_max_mib": None,
            "system_ram_mean_percent": None,
            "system_ram_max_percent": None,
            "gpu_allocated_max_mib": None,
            "gpu_reserved_max_mib": None,
        }

    df = pd.DataFrame(records)

    return {
        "samples": len(df),
        "process_cpu_mean_percent": float(
            df["process_cpu_percent"].mean()
        ),
        "process_cpu_max_percent": float(
            df["process_cpu_percent"].max()
        ),
        "process_rss_mean_mib": float(
            df["process_rss_mib"].mean()
        ),
        "process_rss_max_mib": float(
            df["process_rss_mib"].max()
        ),
        "system_ram_mean_percent": float(
            df["system_ram_percent"].mean()
        ),
        "system_ram_max_percent": float(
            df["system_ram_percent"].max()
        ),
        "gpu_allocated_max_mib": float(
            df["gpu_allocated_mib"].max()
        ),
        "gpu_reserved_max_mib": float(
            df["gpu_reserved_mib"].max()
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    test_path = project_root / "data" / "processed" / "test.csv"
    model_dir = (
        project_root
        / "outputs"
        / "models"
        / "mailpyme_beto_model_v2"
    )
    metrics_dir = project_root / "outputs" / "metrics"
    plots_dir = project_root / "outputs" / "plots"
    logs_dir = project_root / "logs"

    for directory in [
        metrics_dir,
        plots_dir,
        logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(
            f"No se encontró test.csv en: {test_path}"
        )

    if not model_dir.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_dir}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Se cancela el benchmark."
        )

    print("=" * 76)
    print("MAILPYME AI - BENCHMARK DE LATENCIA Y ESTABILIDAD")
    print("=" * 76)
    print(f"Modelo: {model_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Warmup: {WARMUP_RUNS}")
    print(f"Inferencias individuales: {INDIVIDUAL_RUNS}")
    print(f"Pruebas de estabilidad: {STABILITY_RUNS}")
    print(f"Batch sizes: {BATCH_SIZES}")

    test_df = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
    )

    if "text" not in test_df.columns:
        raise ValueError(
            "test.csv no contiene la columna 'text'."
        )

    texts = (
        test_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .tolist()
    )

    if not texts or any(not text for text in texts):
        raise ValueError(
            "Existen textos vacíos en test.csv."
        )

    print("\n1. Cargando tokenizador y modelo local...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        use_fast=True,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    print("2. Ejecutando calentamiento de GPU...")

    warmup_text = texts[0]
    warmup_encoded = prepare_batch(
        [warmup_text],
        tokenizer,
        device,
    )

    for _ in range(WARMUP_RUNS):
        _ = infer_batch(
            warmup_encoded,
            model,
        )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    monitor = ResourceMonitor(interval_seconds=0.05)
    monitor.start()

    benchmark_start = time.perf_counter()

    try:
        print("3. Midiendo latencia individual...")
        latency_df = benchmark_individual_latency(
            texts,
            tokenizer,
            model,
            device,
        )

        print("4. Midiendo throughput por tamaño de lote...")
        throughput_df = benchmark_throughput(
            texts,
            tokenizer,
            model,
            device,
        )

        print("5. Ejecutando prueba de estabilidad...")
        stability_result = run_stability_test(
            texts,
            tokenizer,
            model,
            device,
        )

    finally:
        torch.cuda.synchronize()
        monitor.stop()

    total_elapsed_seconds = time.perf_counter() - benchmark_start
    resource_summary = summarize_resources(
        monitor.records
    )

    latency_values = latency_df["latency_ms"].tolist()

    latency_summary = {
        "runs": len(latency_df),
        "mean_ms": statistics.mean(latency_values),
        "median_ms": statistics.median(latency_values),
        "std_ms": statistics.pstdev(latency_values),
        "min_ms": min(latency_values),
        "p90_ms": percentile(latency_values, 90),
        "p95_ms": percentile(latency_values, 95),
        "p99_ms": percentile(latency_values, 99),
        "max_ms": max(latency_values),
    }

    peak_gpu_allocated_mib = bytes_to_mib(
        torch.cuda.max_memory_allocated()
    )
    peak_gpu_reserved_mib = bytes_to_mib(
        torch.cuda.max_memory_reserved()
    )

    result = {
        "status": "OK",
        "timestamp_local": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "model_dir": str(model_dir),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_compiled": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_vram_mib": bytes_to_mib(
                torch.cuda.get_device_properties(0).total_memory
            ),
            "logical_cpu_count": psutil.cpu_count(
                logical=True
            ),
            "physical_cpu_count": psutil.cpu_count(
                logical=False
            ),
            "system_ram_total_mib": bytes_to_mib(
                psutil.virtual_memory().total
            ),
        },
        "benchmark_config": {
            "max_length": MAX_LENGTH,
            "warmup_runs": WARMUP_RUNS,
            "individual_runs": INDIVIDUAL_RUNS,
            "stability_runs": STABILITY_RUNS,
            "batch_sizes": BATCH_SIZES,
            "batch_repetitions": BATCH_REPETITIONS,
        },
        "individual_latency": latency_summary,
        "throughput": throughput_df.to_dict(
            orient="records"
        ),
        "stability": stability_result,
        "resources": {
            **resource_summary,
            "torch_peak_gpu_allocated_mib": (
                peak_gpu_allocated_mib
            ),
            "torch_peak_gpu_reserved_mib": (
                peak_gpu_reserved_mib
            ),
        },
        "total_benchmark_elapsed_seconds": (
            total_elapsed_seconds
        ),
        "interpretation_note": (
            "Los resultados corresponden a una RTX 4060 Laptop y "
            "no representan necesariamente el rendimiento en CPU, "
            "servidores externos o equipos de producción."
        ),
    }

    latency_csv_path = (
        metrics_dir
        / "latencia_individual_beto_v2.csv"
    )
    throughput_csv_path = (
        metrics_dir
        / "throughput_beto_v2.csv"
    )
    resource_csv_path = (
        metrics_dir
        / "monitoreo_recursos_beto_v2.csv"
    )
    summary_json_path = (
        metrics_dir
        / "benchmark_beto_v2.json"
    )
    stability_csv_path = (
        metrics_dir
        / "estabilidad_beto_v2.csv"
    )
    log_path = (
        logs_dir
        / "benchmark_beto_v2.log"
    )

    latency_df.to_csv(
        latency_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    throughput_df.to_csv(
        throughput_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(monitor.records).to_csv(
        resource_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [stability_result]
    ).to_csv(
        stability_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(result),
            file,
            ensure_ascii=False,
            indent=2,
        )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "MAILPYME AI - BENCHMARK BETO V2\n"
        )
        file.write(
            f"Fecha: {result['timestamp_local']}\n"
        )
        file.write(
            f"GPU: {result['environment']['gpu']}\n"
        )
        file.write(
            f"Latencia media: "
            f"{latency_summary['mean_ms']:.4f} ms\n"
        )
        file.write(
            f"Latencia p95: "
            f"{latency_summary['p95_ms']:.4f} ms\n"
        )
        file.write(
            f"Latencia máxima: "
            f"{latency_summary['max_ms']:.4f} ms\n"
        )
        file.write(
            f"Estabilidad: "
            f"{stability_result['success_rate'] * 100:.2f}%\n"
        )
        file.write(
            f"Excepciones: "
            f"{stability_result['exceptions']}\n"
        )
        file.write(
            f"Pico VRAM asignada: "
            f"{peak_gpu_allocated_mib:.2f} MiB\n"
        )
        file.write(
            f"Tiempo total benchmark: "
            f"{total_elapsed_seconds:.4f} s\n"
        )

    latency_plot_path = (
        plots_dir
        / "distribucion_latencia_beto_v2.png"
    )
    throughput_plot_path = (
        plots_dir
        / "throughput_por_batch_beto_v2.png"
    )

    plt.figure(figsize=(10, 6))
    plt.hist(
        latency_values,
        bins=30,
        edgecolor="black",
        alpha=0.8,
    )
    plt.axvline(
        latency_summary["mean_ms"],
        linestyle="--",
        linewidth=2,
        label=(
            f"Media: "
            f"{latency_summary['mean_ms']:.2f} ms"
        ),
    )
    plt.axvline(
        latency_summary["p95_ms"],
        linestyle=":",
        linewidth=2,
        label=(
            f"P95: "
            f"{latency_summary['p95_ms']:.2f} ms"
        ),
    )
    plt.title(
        "Distribución de latencia individual - BETO MailPyme AI v2"
    )
    plt.xlabel("Latencia por correo (ms)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        latency_plot_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        throughput_df["batch_size"],
        throughput_df[
            "throughput_emails_per_second"
        ],
        marker="o",
    )
    plt.title(
        "Throughput por tamaño de lote - BETO MailPyme AI v2"
    )
    plt.xlabel("Tamaño de lote")
    plt.ylabel("Correos por segundo")
    plt.xticks(BATCH_SIZES)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        throughput_plot_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nResultados de latencia individual:")
    print(
        f"  Media:    {latency_summary['mean_ms']:.4f} ms"
    )
    print(
        f"  Mediana:  {latency_summary['median_ms']:.4f} ms"
    )
    print(
        f"  P90:      {latency_summary['p90_ms']:.4f} ms"
    )
    print(
        f"  P95:      {latency_summary['p95_ms']:.4f} ms"
    )
    print(
        f"  P99:      {latency_summary['p99_ms']:.4f} ms"
    )
    print(
        f"  Máxima:   {latency_summary['max_ms']:.4f} ms"
    )

    print("\nThroughput:")
    print(
        throughput_df[
            [
                "batch_size",
                "mean_batch_latency_ms",
                "mean_latency_per_email_ms",
                "throughput_emails_per_second",
            ]
        ].to_string(index=False)
    )

    print("\nEstabilidad:")
    print(
        f"  Ejecuciones:          "
        f"{stability_result['runs']}"
    )
    print(
        f"  Tasa de éxito:        "
        f"{stability_result['success_rate'] * 100:.2f}%"
    )
    print(
        f"  Inconsistencias:      "
        f"{stability_result['inconsistencies']}"
    )
    print(
        f"  Salidas no finitas:   "
        f"{stability_result['non_finite_outputs']}"
    )
    print(
        f"  Excepciones:          "
        f"{stability_result['exceptions']}"
    )

    print("\nRecursos:")
    print(
        f"  CPU proceso media:    "
        f"{resource_summary['process_cpu_mean_percent']:.2f}%"
    )
    print(
        f"  CPU proceso máxima:   "
        f"{resource_summary['process_cpu_max_percent']:.2f}%"
    )
    print(
        f"  RAM proceso máxima:   "
        f"{resource_summary['process_rss_max_mib']:.2f} MiB"
    )
    print(
        f"  VRAM asignada pico:   "
        f"{peak_gpu_allocated_mib:.2f} MiB"
    )
    print(
        f"  VRAM reservada pico:  "
        f"{peak_gpu_reserved_mib:.2f} MiB"
    )

    print("\nArchivos generados:")
    for path in [
        latency_csv_path,
        throughput_csv_path,
        resource_csv_path,
        stability_csv_path,
        summary_json_path,
        log_path,
        latency_plot_path,
        throughput_plot_path,
    ]:
        print(f"  {path}")

    print(
        "\nBENCHMARK DE LATENCIA Y ESTABILIDAD COMPLETADO CORRECTAMENTE."
    )


if __name__ == "__main__":
    main()
