import argparse
import json
import re
from pathlib import Path


def extract_statistic(md_path: Path, statistic: str):
    pattern = re.compile(rf"\|\s*{re.escape(statistic)}\s*\|\s*([0-9.]+)\s*\|")

    with md_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))

    return None


def extract_inference_minutes(md_path: Path):
    """
    Extract inference time in minutes from a line like:
    Ensemble (process_images): 237.60 s (3.96 min)
    """
    pattern = re.compile(r"Ensemble \(process_images\):\s*[\d.]+\s*s\s*\(([\d.]+)\s*min\)")
    with md_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None


def extract_model_indices(filename: str):
    """
    Extract model indices from filenames like:
    change_detection_5_models_2026_SOTA_subset_1_3_4_parcellhuse_96117.md
    """
    m = re.search(r"_subset_([0-9_]+)_", filename)
    if not m:
        return []

    return [int(x) for x in m.group(1).split("_")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        default="/mnt/T/mnt/ML_output/building_change_detection_2026/evaluations",
    )
    parser.add_argument(
        "--area",
        default="parcellhuse",
    )
    parser.add_argument(
        "--output_directory",
        default="/mnt/T/mnt/ML_output/building_change_detection_2026/evaluations/",
    )
    parser.add_argument(
        "--statistic",
        default="Pixel accuracy",
    )
    parser.add_argument(
        "--original_config",
        default="config_files/change_detection_5_models_2026_SOTA.json",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load config
    with open(args.original_config, "r") as f:
        config = json.load(f)

    model_names = config.get("model_names", [])

    md_files = sorted(folder.glob(f"*{args.area}*.md"))

    results = []
    used_model_indices = set()

    for md_file in md_files:
        value = extract_statistic(md_file, args.statistic)
        if value is None:
            continue

        inference_min = extract_inference_minutes(md_file)
        indices = extract_model_indices(md_file.name)
        used_model_indices.update(indices)

        results.append((value, inference_min, md_file.name))

    results.sort(key=lambda x: x[0], reverse=True)

    safe_stat = args.statistic.replace(" ", "_")
    output_path = output_dir / f"{args.area}-{safe_stat}.md"

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# Files sorted by {args.statistic}\n\n")
        f.write("| Score | Inference (min) | File |\n")
        f.write("|-------|----------------|------|\n")

        for value, inference_min, filename in results:
            time_str = f"{inference_min:.2f}" if inference_min is not None else "—"
            f.write(f"| {value:.6f} | {time_str} | {filename} |\n")

        f.write("\n---\n")
        f.write("\n# Model index mapping\n\n")

        for idx in sorted(used_model_indices):
            if idx < len(model_names):
                f.write(f"{idx}: {model_names[idx]}\n")
            else:
                f.write(f"{idx}: UNKNOWN\n")

    print(f"Wrote {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
