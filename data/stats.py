import json
import statistics
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "dataset.txt"


def read_data(path: Path = DATASET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def stage_lengths(dataset: list[dict]) -> dict[str, list[int]]:
    lengths = {}
    for data in dataset:
        for stage, moves in data["solution"].items():
            lengths.setdefault(stage, [])
            # empty stage: already done before it started, nothing to count
            if moves.strip():
                lengths[stage].append(len(moves.split()))
    return lengths


if __name__ == "__main__":
    dataset = read_data()
    lengths = stage_lengths(dataset)

    print(f"{len(dataset)} scrambles\n")
    print(f"{'stage':<16}{'used':>7}{'empty':>7}{'mean':>8}{'median':>8}{'min':>6}{'max':>6}")
    for stage, values in lengths.items():
        empty = len(dataset) - len(values)
        if not values:
            print(f"{stage:<16}{0:>7}{empty:>7}{'-':>8}{'-':>8}{'-':>6}{'-':>6}")
            continue
        print(
            f"{stage:<16}{len(values):>7}{empty:>7}"
            f"{statistics.mean(values):>8.1f}{statistics.median(values):>8.1f}"
            f"{min(values):>6}{max(values):>6}"
        )
