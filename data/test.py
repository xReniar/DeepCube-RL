import json
from magiccube import Cube


DATASET_PATH = "dataset.txt"


def read_data(path: str = DATASET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


dataset = read_data()

for i, data in enumerate(dataset):
    # initial_state is the cube after the scramble has been applied
    cube = Cube(state=data["initial_state"])

    print("=" * 40, f" scramble {i + 1}/{len(dataset)} ", "=" * 40)
    print(cube)

    for step, moves in data["solution"].items():
        print("#" * 10, f" {step} ", "#" * 10)

        for move in moves.split():
            cube.rotate(move)
            print(f"move: {move}")
            print(cube)

    print("solved" if cube.is_done() else "NOT solved")
    input("Press Enter for the next scramble...")
