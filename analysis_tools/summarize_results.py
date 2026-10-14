"""Build readable results and compact CSVs from benchmark records (no GPU needed)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from harness.campaign import aggregate, worker_records

WORKLOADS = ("short", "medium", "long")
IMPLEMENTATIONS = {
    "flash_tiled": "Flash (tiled)",
    "flash_untiled": "Flash (untiled)",
    "apex": "Apex",
    "numba": "Numba",
    "optimized": "Optimized Transducer",
}
COMPONENTS = {
    "untiled_compiled": ("Untiled", "Compiled"),
    "untiled_specialized_joint": ("Untiled", "Specialized joint"),
    "untiled_specialized_normalization": ("Untiled", "Specialized normalization"),
    "untiled_specialized": ("Untiled", "Both specialized"),
    "tiled_compiled": ("Tiled", "Compiled"),
    "tiled_specialized_joint": ("Tiled", "Specialized joint"),
    "tiled_specialized_normalization": ("Tiled", "Specialized normalization"),
    "tiled_specialized": ("Tiled", "Both specialized"),
    "untiled_whole_graph": ("Untiled", "Whole-graph compilation"),
    "tiled_whole_graph": ("Tiled", "Whole-graph compilation"),
}


def in_panel(row, schedule):
    """Full-batch runs, including tiled Flash, appear in both schedule panels."""
    return row["config"]["schedule"] in (schedule, "full_batch")


def select_within_peak(rows, tolerance):
    good = [r for r in rows if r["status"] == "complete"]
    if not good:
        return None
    peak = max(r["samples_per_second"] for r in good)
    return min(
        (r for r in good if r["samples_per_second"] >= (1 - tolerance) * peak),
        key=lambda r: (r["allocated_bytes"], r["median_ms"]),
    )


def verify_campaign(folder):
    """Rebuild the summary from the worker records and check it matches the stored one."""
    folder = Path(folder)
    manifest_bytes = (folder / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads((folder / "summary.json").read_text())
    if hashlib.sha256(manifest_bytes).hexdigest() != summary["manifest_sha256"]:
        raise ValueError(f"{folder}: manifest checksum mismatch")
    rebuilt = aggregate(list(worker_records(folder)), manifest)

    def key(row):
        return row["workload"], row["config"]["id"]

    if sorted(rebuilt, key=key) != sorted(summary["cells"], key=key):
        raise ValueError(f"{folder}: summary differs from raw records")
    if any(row["status"] not in ("complete", "oom") for row in rebuilt):
        raise ValueError(
            f"{folder}: unresolved or failed measurement; inspect the raw records"
        )
    return manifest, rebuilt


def measurements(cell):
    # CSVs keep full precision; only the Markdown tables are rounded.
    return {
        "samples_per_second": cell.get("samples_per_second", ""),
        "peak_memory_gib": cell["allocated_bytes"] / 2**30
        if cell["status"] == "complete"
        else "",
        "status": "ok" if cell["status"] == "complete" else "OOM",
    }


def comparison_rows(cells):
    def order(cell):
        c = cell["config"]
        return (
            WORKLOADS.index(cell["workload"]),
            tuple(IMPLEMENTATIONS).index(c["family"]),
            ("full_batch", "retain", "release").index(c["schedule"]),
            c["tile_rows"] or c["sub_batch"] or 0,
        )

    for cell in sorted(cells, key=order):
        c = cell["config"]
        yield {
            "workload": cell["workload"],
            "implementation": IMPLEMENTATIONS[c["family"]],
            "schedule": c["schedule"].replace("_", " "),
            "sub_batch_size": c["sub_batch"] or "",
            "tile_rows": c["tile_rows"] or "",
            **measurements(cell),
        }


def ablation_rows(cells):
    for cell in sorted(
        cells,
        key=lambda r: (
            WORKLOADS.index(r["workload"]),
            tuple(COMPONENTS).index(r["config"]["id"]),
        ),
    ):
        execution, components = COMPONENTS[cell["config"]["id"]]
        yield {
            "workload": cell["workload"],
            "execution": execution,
            "components": components,
            **measurements(cell),
        }


def configuration(cell):
    c = cell["config"]
    if c["family"] == "flash_tiled":
        return f"Tile {c['tile_rows']:,} rows"
    if c["schedule"] == "full_batch":
        return "Full batch"
    return f"Sub-batch {c['sub_batch']}"


def display(cell):
    if cell["status"] == "oom":
        return "OOM", "—"
    speed = f"{cell['samples_per_second']:,.{1 if cell['workload'] == 'long' else 0}f}"
    return speed, f"{cell['allocated_bytes'] / 2**30:.2f}"


def summary_text(comparison, ablation, comparison_manifest, ablation_manifest):
    m = comparison_manifest
    tile_rows = {
        c["tile_rows"] for c in ablation_manifest["configurations"] if c["tiled"]
    }
    if len(tile_rows) != 1:
        raise ValueError("Tiled ablation variants must share one tile size")
    tile_rows = next(iter(tile_rows))
    devices = {d["name"] for d in m["environment"]["devices"]}
    text = [
        "# Results",
        "",
        f"{', '.join(sorted(devices))}; B{m['batch']}, H{m['hidden']}, V{m['vocab']}, "
        f"dropout {m['dropout_p']}. Projected-input joint-and-loss forward/backward.",
        "",
        f"All tested configurations, including OOMs: [comparison.csv](comparison.csv) "
        f"({len(comparison)} rows) and [ablation.csv](ablation.csv) ({len(ablation)} rows).",
        "",
        "## Comparison",
        "",
        "Each entry selects the lowest-memory configuration within 5% of that "
        "implementation’s peak throughput in the indicated schedule panel. "
        "Tiled Flash and full-batch measurements are available in both panels.",
        "",
    ]
    for schedule in ("retain", "release"):
        text += [
            f"### {schedule.title()}",
            "",
            "| Workload | Implementation | Selected configuration | Samples/s | Peak GiB |",
            "|---|---|---|---:|---:|",
        ]
        for name in WORKLOADS:
            for family, label in IMPLEMENTATIONS.items():
                candidates = [
                    r
                    for r in comparison
                    if r["workload"] == name
                    and r["config"]["family"] == family
                    and in_panel(r, schedule)
                ]
                if not candidates:
                    raise ValueError(f"Missing comparison: {name}/{family}/{schedule}")
                chosen = select_within_peak(candidates, 0.05)
                setting = configuration(chosen) if chosen else "All OOM"
                speed, memory = display(chosen) if chosen else ("—", "—")
                text.append(
                    f"| {name.title()} | {label} | {setting} | {speed} | {memory} |"
                )
        text.append("")
    text += [
        "## Ablation",
        "",
        f"Samples/s / peak GiB. Tiled variants use {tile_rows:,} rows per tile; "
        "untiled variants use the full batch without sub-batching.",
        "",
        "| Execution | Components | Short | Medium | Long |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, (execution, components) in COMPONENTS.items():
        values = []
        for name in WORKLOADS:
            matches = [
                r
                for r in ablation
                if r["workload"] == name and r["config"]["id"] == variant
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one ablation result: {name}/{variant}")
            row = matches[0]
            speed, memory = display(row)
            values.append("OOM" if row["status"] == "oom" else f"{speed} / {memory}")
        text.append("| " + " | ".join([execution, components, *values]) + " |")
    text += [
        "",
        "## Files and regeneration",
        "",
        "CSV columns: workload, implementation or components, execution settings, "
        "throughput, peak allocated memory and status. `full batch` means one backward "
        "pass without sub-batching; tile rows apply only to tiled Flash.",
        "",
        "Run `make results RESULTS=<campaign-folder>` from the repository root "
        "to check the records and regenerate these tables and CSVs. "
        "Manifests, summaries and worker records, including timing warnings, are under "
        "`raw/comparison/` and `raw/ablation/`.",
        "",
    ]
    return "\n".join(text)


def summarize(comparison_folder, ablation_folder, output):
    comparison_manifest, comparison = verify_campaign(comparison_folder)
    ablation_manifest, ablation = verify_campaign(ablation_folder)
    workloads = hashlib.sha256((ROOT / "data/workloads/workloads.json").read_bytes())
    if {
        comparison_manifest["workload_sha256"],
        ablation_manifest["workload_sha256"],
    } != {workloads.hexdigest()}:
        raise ValueError(
            "Records were measured on other workloads than data/workloads/"
        )
    text = summary_text(comparison, ablation, comparison_manifest, ablation_manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("comparison", list(comparison_rows(comparison))),
        ("ablation", list(ablation_rows(ablation))),
    ):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} configurations")
    (output / "README.md").write_text(text)
    print(f"Summary: {output / 'README.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", type=Path, default=ROOT / "results/local/raw/comparison"
    )
    parser.add_argument("--ablation", type=Path, default=ROOT / "results/local/raw/ablation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/local")
    args = parser.parse_args()
    summarize(args.comparison, args.ablation, args.output)


if __name__ == "__main__":
    main()
