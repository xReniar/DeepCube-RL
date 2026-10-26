import argparse
import contextlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from magiccube import Cube

from data import conditions
from models.direct_state_policy import (
    ACTIONS, CONTEXT, DEVICE as MODEL_DEVICE, INPUT_VOCAB, MOVES, DirectStatePolicy
)

DEVICE = torch.device(MODEL_DEVICE)

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "dataset.txt"
OUTPUT_DIR = ROOT / "checkpoints"
CHECKPOINT_NAME = "checkpoint.pt"
WEIGHTS_NAME = "weights.pt"

# quarter turns only, the solver in dataset.txt never uses double turns
MOVE_NAMES = [face + turn for face in "URFDLB" for turn in ("", "'")]
MOVE_ID = {name: index for index, name in enumerate(MOVE_NAMES)}
CENTERS = (4, 13, 22, 31, 40, 49)

# stages in the order they are solved, same names as the keys in dataset.txt
STAGES = (
    "d_cross", "first_layer", "second_layer",
    "u_cross", "permute_edges", "permute_corners"
)
# after the 12 moves come the stage tokens: the model emits <stage>_start before the
# moves of a stage and <stage>_end when the stage is done
STAGE_START = {name: MOVES + 2 * index for index, name in enumerate(STAGES)}
STAGE_END = {name: MOVES + 2 * index + 1 for index, name in enumerate(STAGES)}
TOKEN_NAMES = MOVE_NAMES + [
    f"<|{name}_{kind}|>" for name in STAGES for kind in ("start", "end")
]
BOS = ACTIONS

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_NORM = 1.0
VAL_FRACTION = 0.05
EVAL_EVERY = 5
EVAL_ROWS = 64
SEED = 49_049

USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

assert len(MOVE_NAMES) == MOVES and 2 * len(STAGES) == ACTIONS - MOVES
assert INPUT_VOCAB == ACTIONS + 1


def autocast_context():
    if USE_BF16:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_s54(states: list[str]) -> np.ndarray:
    """Color letters -> 0..5 (index of the face that owns the color), shape (n, 54).

    Face moves never move the centers, so the table of the first state is valid
    for every state of the same cube.
    """
    table = np.zeros(256, dtype=np.uint8)
    for face, position in enumerate(CENTERS):
        table[ord(states[0][position])] = face
    codes = np.frombuffer("".join(states).encode(), dtype=np.uint8)
    return table[codes].reshape(len(states), 54)


def stage_done(cube: Cube, stage: str) -> bool:
    return getattr(conditions, stage)(cube)


def build_example(row: dict, stage: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Moves of every stage up to (and including) `stage`, between its start and end tokens.

    Stages with no moves are skipped (no tokens). Returns None if there is nothing left.
    """
    cube = Cube(state=row["initial_state"])
    states, target = [], []
    for name in STAGES[:STAGES.index(stage) + 1]:
        moves = row["solution"][name].split()
        if not moves:
            continue
        # the input of every token is the cube before it, stage tokens don't move the cube
        states.append(cube.get())
        target.append(STAGE_START[name])
        for move in moves:
            states.append(cube.get())
            target.append(MOVE_ID[move])
            cube.rotate(move)
        assert stage_done(cube, name)
        states.append(cube.get())
        target.append(STAGE_END[name])

    if not target:
        return None
    return torch.from_numpy(to_s54(states)), torch.tensor(target)


def load_dataset(path: Path, stage: str, limit: int | None = None):
    with open(path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:limit]

    examples, initial_states = [], []
    empty = too_long = 0
    for row in rows:
        example = build_example(row, stage)
        if example is None:
            empty += 1
            continue
        if len(example[1]) > CONTEXT:
            too_long += 1
            continue
        examples.append(example)
        initial_states.append(row["initial_state"])

    lengths = [len(target) for _, target in examples]
    print(
        f"dataset | until {stage} | {len(examples):,}/{len(rows):,} scrambles | "
        f"tokens min/mean/max {min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)} | "
        f"{empty:,} already done | {too_long:,} dropped (> {CONTEXT} tokens)"
    )
    return examples, initial_states


@dataclass
class Batch:
    token: torch.Tensor
    s54: torch.Tensor
    target: torch.Tensor
    valid: torch.Tensor
    length: torch.Tensor

    def __len__(self):
        return len(self.token)


def make_batch(examples) -> Batch:
    length = torch.tensor([len(target) for _, target in examples])
    width = int(length.max())
    s54 = torch.zeros(len(examples), width, 54, dtype=torch.long)
    target = torch.zeros(len(examples), width, dtype=torch.long)
    for row, (state, moves) in enumerate(examples):
        s54[row, :len(moves)] = state
        target[row, :len(moves)] = moves

    valid = torch.arange(width)[None] < length[:, None]
    token = torch.full_like(target, BOS)
    token[:, 1:] = target[:, :-1]
    return Batch(*(
        tensor.to(DEVICE) for tensor in (token, s54, target, valid, length)
    ))


def policy_loss(logits, batch: Batch):
    token_ce = F.cross_entropy(
        logits.float().flatten(0, 1), batch.target.flatten(), reduction="none"
    ).view_as(batch.target)
    per_sequence = (token_ce * batch.valid).sum(1) / batch.length
    return per_sequence.mean(), per_sequence


def training_step(model, optimizer, batch: Batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context():
        logits = model(batch.token, batch.s54)
    loss, per_sequence = policy_loss(logits, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
    optimizer.step()
    return per_sequence.detach(), logits.argmax(-1).eq(batch.target) & batch.valid


@torch.no_grad()
def validate(model, examples, batch_size):
    model.eval()
    loss_sum = correct = labels = 0.0
    for start in range(0, len(examples), batch_size):
        batch = make_batch(examples[start:start + batch_size])
        with autocast_context():
            logits = model(batch.token, batch.s54)
        _, per_sequence = policy_loss(logits, batch)
        loss_sum += float(per_sequence.sum())
        correct += float((logits.argmax(-1).eq(batch.target) & batch.valid).sum())
        labels += float(batch.valid.sum())
    return loss_sum / len(examples), correct / labels


@torch.no_grad()
def greedy_solve(model, initial_states: list[str], stage: str):
    """Greedy rollout: the model picks a token, magiccube applies the moves.

    A cube is finished when the model emits an end token and `stage` is done (or the
    token is the end of `stage`): it counts as solved only if `stage` is done.
    """
    model.eval()
    cubes = [Cube(state=state) for state in initial_states]
    finished = [False] * len(cubes)
    solved = [False] * len(cubes)
    cache = model.new_cache(
        len(cubes), torch.bfloat16 if USE_BF16 else torch.float32
    )
    token = torch.full((len(cubes), 1), BOS, dtype=torch.long, device=DEVICE)
    for step in range(CONTEXT):
        s54 = torch.from_numpy(to_s54([cube.get() for cube in cubes]))
        with autocast_context():
            logits = model(token, s54[:, None].long().to(DEVICE), cache)[:, -1]
        action = logits.argmax(-1)
        for index, move in enumerate(action.tolist()):
            if finished[index]:
                continue
            if move < MOVES:
                cubes[index].rotate(MOVE_NAMES[move])
            elif move in STAGE_END.values():
                solved[index] = stage_done(cubes[index], stage)
                finished[index] = solved[index] or move == STAGE_END[stage]
        if all(finished):
            break
        token = action[:, None]
    return sum(solved), len(cubes)


def save_checkpoint(path, model, optimizer, epoch, best_loss):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, temporary)
    temporary.replace(path)


def train(
    stage: str,
    dataset_path: Path = DATASET_PATH,
    output_dir: Path | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    eval_every: int = EVAL_EVERY,
    eval_rows: int = EVAL_ROWS,
    limit: int | None = None,
    resume: bool = False,
):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")

    examples, initial_states = load_dataset(dataset_path, stage, limit)
    order = list(range(len(examples)))
    random.Random(SEED).shuffle(order)
    val_size = max(1, int(len(order) * VAL_FRACTION))
    val_index, train_index = order[:val_size], order[val_size:]
    train_examples = [examples[i] for i in train_index]
    val_examples = [examples[i] for i in val_index]
    val_states = [initial_states[i] for i in val_index][:eval_rows]
    print(f"split | {len(train_examples):,} train | {len(val_examples):,} validation")

    model = DirectStatePolicy().to(DEVICE)
    kwargs = dict(lr=learning_rate, weight_decay=WEIGHT_DECAY)
    if DEVICE.type == "cuda":
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **kwargs)

    output_dir = output_dir or OUTPUT_DIR / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / CHECKPOINT_NAME
    weights_path = output_dir / WEIGHTS_NAME
    start_epoch, best_loss = 0, float("inf")
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
        print(f"checkpoint resumed | epoch {start_epoch}")

    print(
        f"DirectStatePolicy | params {model.parameter_count:,} | device {DEVICE} | "
        f"bf16 {USE_BF16} | batch {batch_size} | lr {learning_rate:.1e}"
    )

    started = time.perf_counter()
    for epoch in range(start_epoch + 1, epochs + 1):
        random.shuffle(train_examples)
        loss_sum = correct = labels = 0.0
        for start in range(0, len(train_examples), batch_size):
            batch = make_batch(train_examples[start:start + batch_size])
            per_sequence, hits = training_step(model, optimizer, batch)
            loss_sum += float(per_sequence.sum())
            correct += float(hits.sum())
            labels += float(batch.valid.sum())

        val_loss, val_accuracy = validate(model, val_examples, batch_size)
        message = (
            f"epoch {epoch}/{epochs} | loss {loss_sum / len(train_examples):.4f} | "
            f"acc {100 * correct / labels:.1f}% | val loss {val_loss:.4f} | "
            f"val acc {100 * val_accuracy:.1f}%"
        )
        if epoch % eval_every == 0 or epoch == epochs:
            solved, total = greedy_solve(model, val_states, stage)
            message += f" | greedy solved {solved}/{total}"
        elapsed = (time.perf_counter() - started) / 60
        print(f"{message} | {elapsed:.1f} min")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, weights_path)
        save_checkpoint(checkpoint_path, model, optimizer, epoch, best_loss)

    print(f"training complete | best val loss {best_loss:.4f} | weights {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True, help="solve until this stage (included)")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: checkpoints/<stage>")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-rows", type=int, default=EVAL_ROWS)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N scrambles")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        stage=args.stage,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        eval_rows=args.eval_rows,
        limit=args.limit,
        resume=args.resume,
    )
