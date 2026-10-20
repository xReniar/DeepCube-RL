import random
import hashlib
from multiprocessing import Pool, cpu_count
from magiccube import Cube

MOVES = ["U", "U'", "D", "D'", "L", "L'", "R", "R'", "F", "F'", "B", "B'"]

def _inverse(m: str) -> str:
    return m[0] if m.endswith("'") else m + "'"

def _allowed(seq: list[str], m: str) -> bool:
    """Evita mosse ovviamente ridondanti (R R' e R R R)."""
    if not seq:
        return True
    if m == _inverse(seq[-1]):
        return False
    if len(seq) >= 2 and m == seq[-1] == seq[-2]:
        return False
    return True

def _key(state: str) -> bytes:
    """Hash a 16 byte dello stato, per risparmiare memoria nel set."""
    return hashlib.blake2b(state.encode(), digest_size=16).digest()

def _exhaustive(max_depth: int, seen: set) -> list[tuple[int, str]]:
    frontier = [(Cube(3).get(), [])]
    rows = []
    for depth in range(1, max_depth + 1):
        nxt = []
        for state, path in frontier:
            for m in MOVES:
                cube = Cube(state=state)
                cube.rotate(m)
                new_state = cube.get()
                k = _key(new_state)
                if k in seen:
                    continue
                seen.add(k)
                nxt.append((new_state, path + [m]))
        rows.extend(" ".join(p) for _, p in nxt)
        frontier = nxt
        print(f"[BFS] depth {depth}: {len(nxt)} scrambles")
    return rows

def _worker(args):
    length, n, seed = args
    rng = random.Random(seed)
    cube = Cube(3)
    out = {}
    for _ in range(n):
        seq = []
        while len(seq) < length:
            m = rng.choice(MOVES)
            if _allowed(seq, m):
                seq.append(m)
        scramble = " ".join(seq)
        cube.reset()
        cube.rotate(scramble)
        out[_key(cube.get())] = scramble
    return list(out.items())

def _sample_length(pool, length, target, seen, n_workers, batch=5000):
    rows, stale = [], 0
    while len(rows) < target and stale < 5:
        before = len(rows)
        tasks = [(length, batch, random.getrandbits(64)) for _ in range(n_workers)]
        for items in pool.imap_unordered(_worker, tasks):
            for k, s in items:
                if k in seen:
                    continue
                seen.add(k)
                rows.append(s)
                if len(rows) >= target:
                    break
            if len(rows) >= target:
                break
        stale = stale + 1 if len(rows) == before else 0
    print(f"[SAMPLE] length {length}: {len(rows)} scrambles")
    return rows


def generate_scrambles(
    max_length: int = 5,
    exhaustive_depth: int = 5,
    samples_per_length: int = 10000 ,
) -> list[str]:
    seen = {_key(Cube(3).get())}
    rows = _exhaustive(exhaustive_depth, seen)

    n_workers = cpu_count()
    with Pool(n_workers) as pool:
        for length in range(exhaustive_depth + 1, max_length + 1):
            rows += _sample_length(pool, length, samples_per_length, seen, n_workers)
    return rows