from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def _load_grouped(raw_csv: Path):
    grouped = defaultdict(list)
    if not raw_csv.exists():
        return grouped
    with raw_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            grouped[(row.get("dataset", ""), row.get("model_name", ""), row.get("method", ""))].append(row)
    return grouped


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def _group_means(raw_csv: Path) -> List[Dict[str, object]]:
    rows = []
    for (dataset, model, method), vals in _load_grouped(raw_csv).items():
        if not vals:
            continue
        rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "label": f"{dataset}:{model}:{method.replace('qire_', '')[:18]}",
            "accuracy": mean(_f(v, "accuracy") for v in vals),
            "balanced_accuracy": mean(_f(v, "balanced_accuracy", _f(v, "accuracy")) for v in vals),
            "ops": mean(_f(v, "estimated_ops") for v in vals),
            "time": mean(_f(v, "inference_time_ms") for v in vals),
            "p95": mean(_f(v, "inference_time_ms_p95") for v in vals),
            "memory": mean(_f(v, "memory_mb") for v in vals),
            "energy": mean(_f(v, "estimated_energy_wh") for v in vals),
            "active": mean(_f(v, "active_model_parts") for v in vals),
            "router_ops": mean(_f(v, "router_ops") for v in vals),
            "expert_ops": mean(_f(v, "expert_ops") for v in vals),
            "n": len(vals),
        })
    return rows


def _scatter(rows: List[Dict[str, object]], xkey: str, ykey: str, xlabel: str, ylabel: str, title: str, output_png: Path) -> bool:
    if not rows:
        return False
    fig = plt.figure(figsize=(9.5, 5.8))
    ax = fig.add_subplot(111)
    xs = [float(r[xkey]) for r in rows]
    ys = [float(r[ykey]) for r in rows]
    ax.scatter(xs, ys)
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(str(r["label"]), (x, y), fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def _bar(rows: List[Dict[str, object]], key: str, ylabel: str, title: str, output_png: Path, top_n: int = 24) -> bool:
    if not rows:
        return False
    data = sorted(rows, key=lambda r: float(r.get(key, 0)), reverse=(key in {"accuracy", "balanced_accuracy"}))[:top_n]
    fig = plt.figure(figsize=(max(9.5, len(data) * 0.45), 5.8))
    ax = fig.add_subplot(111)
    labels = [str(r["label"]) for r in data]
    values = [float(r[key]) for r in data]
    ax.bar(range(len(data)), values)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    return True


def plot_accuracy_vs_ops(raw_csv: Path, output_png: Path) -> None:
    rows = _group_means(raw_csv)
    _scatter(rows, "ops", "accuracy", "Estimated ops per sample", "Accuracy", "Accuracy vs estimated operations", output_png)


def plot_all_figures(raw_csv: Path, figures_dir: Path) -> Dict[str, object]:
    rows = _group_means(raw_csv)
    made = []
    specs = [
        ("accuracy_vs_ops.png", lambda: _scatter(rows, "ops", "accuracy", "Estimated ops/sample", "Accuracy", "Accuracy vs estimated operations", figures_dir / "accuracy_vs_ops.png")),
        ("balanced_accuracy_vs_ops.png", lambda: _scatter(rows, "ops", "balanced_accuracy", "Estimated ops/sample", "Balanced accuracy", "Balanced accuracy vs ops", figures_dir / "balanced_accuracy_vs_ops.png")),
        ("accuracy_vs_time.png", lambda: _scatter(rows, "time", "accuracy", "Inference ms/sample", "Accuracy", "Accuracy vs inference time", figures_dir / "accuracy_vs_time.png")),
        ("accuracy_vs_memory.png", lambda: _scatter(rows, "memory", "accuracy", "Memory MB", "Accuracy", "Accuracy vs memory", figures_dir / "accuracy_vs_memory.png")),
        ("accuracy_vs_energy.png", lambda: _scatter(rows, "energy", "accuracy", "Estimated training energy Wh", "Accuracy", "Accuracy vs estimated energy", figures_dir / "accuracy_vs_energy.png")),
        ("inference_time_bar.png", lambda: _bar(rows, "time", "Inference ms/sample", "Inference time by model", figures_dir / "inference_time_bar.png")),
        ("active_parts_bar.png", lambda: _bar(rows, "active", "Active model parts", "Active model parts by model", figures_dir / "active_parts_bar.png")),
        ("router_ops_bar.png", lambda: _bar(rows, "router_ops", "Router ops/sample", "Router overhead by model", figures_dir / "router_ops_bar.png")),
    ]
    for name, fn in specs:
        try:
            if fn():
                made.append(name)
        except Exception:
            continue
    return {"figure_count": len(made), "figures": made}

# v0.6 extended plot functions. They intentionally use simple matplotlib primitives for reliability on Windows.
def _line_timeline(raw_csv: Path, output_png: Path) -> bool:
    rows = []
    if not raw_csv.exists():
        return False
    with raw_csv.open("r", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if row.get("status", "ok") == "ok":
                rows.append((i, _f(row, "balanced_accuracy", _f(row, "accuracy"))))
    if len(rows) < 2:
        return False
    fig = plt.figure(figsize=(9.5, 5.2)); ax = fig.add_subplot(111)
    ax.plot([x for x,_ in rows], [y for _,y in rows])
    ax.set_xlabel("Successful run order"); ax.set_ylabel("Balanced accuracy"); ax.set_title("Research timeline: balanced accuracy over completed runs")
    fig.tight_layout(); output_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_png, dpi=160); plt.close(fig); return True


def _seed_stability(raw_csv: Path, output_png: Path) -> bool:
    grouped = _load_grouped(raw_csv)
    data = []
    for key, vals in grouped.items():
        if len(vals) >= 2:
            accs = [_f(v, "balanced_accuracy", _f(v, "accuracy")) for v in vals]
            mu = mean(accs)
            var = sum((a-mu)**2 for a in accs) / max(1, len(accs)-1)
            data.append((":".join(key), var ** 0.5))
    if not data:
        return False
    data = sorted(data, key=lambda x: x[1], reverse=True)[:24]
    fig = plt.figure(figsize=(max(9.5, len(data)*0.42), 5.8)); ax = fig.add_subplot(111)
    ax.bar(range(len(data)), [v for _,v in data]); ax.set_xticks(range(len(data))); ax.set_xticklabels([k[:36] for k,_ in data], rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Std. dev. balanced accuracy"); ax.set_title("Seed stability risk by model")
    fig.tight_layout(); output_png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(output_png, dpi=160); plt.close(fig); return True


_old_plot_all_figures = plot_all_figures

def plot_all_figures(raw_csv: Path, figures_dir: Path) -> Dict[str, object]:
    rows = _group_means(raw_csv)
    made = []
    specs = [
        ("accuracy_vs_ops.png", lambda: _scatter(rows, "ops", "accuracy", "Estimated ops/sample", "Accuracy", "Accuracy vs estimated operations", figures_dir / "accuracy_vs_ops.png")),
        ("balanced_accuracy_vs_ops.png", lambda: _scatter(rows, "ops", "balanced_accuracy", "Estimated ops/sample", "Balanced accuracy", "Balanced accuracy vs ops", figures_dir / "balanced_accuracy_vs_ops.png")),
        ("accuracy_vs_time.png", lambda: _scatter(rows, "time", "accuracy", "Inference ms/sample", "Accuracy", "Accuracy vs inference time", figures_dir / "accuracy_vs_time.png")),
        ("accuracy_vs_p95_time.png", lambda: _scatter(rows, "p95", "accuracy", "p95 inference ms/sample", "Accuracy", "Accuracy vs p95 inference time", figures_dir / "accuracy_vs_p95_time.png")),
        ("accuracy_vs_memory.png", lambda: _scatter(rows, "memory", "accuracy", "Memory MB", "Accuracy", "Accuracy vs memory", figures_dir / "accuracy_vs_memory.png")),
        ("accuracy_vs_energy.png", lambda: _scatter(rows, "energy", "accuracy", "Estimated training energy Wh", "Accuracy", "Accuracy vs estimated energy", figures_dir / "accuracy_vs_energy.png")),
        ("inference_time_bar.png", lambda: _bar(rows, "time", "Inference ms/sample", "Inference time by model", figures_dir / "inference_time_bar.png")),
        ("model_ranking_accuracy.png", lambda: _bar(rows, "balanced_accuracy", "Balanced accuracy", "Model ranking by balanced accuracy", figures_dir / "model_ranking_accuracy.png")),
        ("active_parts_bar.png", lambda: _bar(rows, "active", "Active model parts", "Active model parts by model", figures_dir / "active_parts_bar.png")),
        ("router_ops_bar.png", lambda: _bar(rows, "router_ops", "Router ops/sample", "Router overhead by model", figures_dir / "router_ops_bar.png")),
        ("expert_ops_bar.png", lambda: _bar(rows, "expert_ops", "Expert ops/sample", "Expert computation by model", figures_dir / "expert_ops_bar.png")),
        ("timeline.png", lambda: _line_timeline(raw_csv, figures_dir / "timeline.png")),
        ("seed_stability.png", lambda: _seed_stability(raw_csv, figures_dir / "seed_stability.png")),
    ]
    for name, fn in specs:
        try:
            if fn(): made.append(name)
        except Exception:
            continue
    return {"figure_count": len(made), "figures": made}
