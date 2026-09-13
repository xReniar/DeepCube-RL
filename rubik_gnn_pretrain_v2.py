"""
Pretraining di una GNN per rappresentare il cubo di Rubik a livello di pezzo.

Motore del cubo: libreria `magiccube` (https://github.com/trincaog/magiccube).
Costruzione del grafo a 54 nodi: `cube2graph` (fornito dall'utente), adattata
per usare feature one-hot invece che uno scalare con segno, e per separare la
topologia (fissa) dai colori (che cambiano ad ogni scramble).

Pipeline:
  1. magiccube.Cube genera stati del cubo (scramble casuali). Gli oggetti
     CubePiece restano fisicamente gli stessi oggetti Python mentre il cubo
     ruota -> usiamo id(piece) per tracciare quale pezzo fisico occupa ogni
     slot: e' il ground truth "gratis" per il pretraining.
  2. Il cubo viene esportato come grafo a 54 nodi con la topologia di
     cube2graph (adiacenza fisica reale sticker-sticker, incluse le
     giunzioni tra facce) e colore one-hot come feature di nodo.
  3. Una GNN (encoder) produce un embedding per sticker.
  4. Un pooling DETERMINISTICO (non learnable) raggruppa gli sticker per
     pezzo fisico -> 26 embedding di "pezzo" (piece-level graph). La mappa
     sticker->pezzo (STICKER_TO_SLOT) e' derivata dalla stessa geometria
     usata da cube2graph, non da un pooling separato "stesso pezzo": in
     questa topologia gli sticker dello stesso pezzo risultano gia'
     naturalmente in una cricca del grafo (verificato).
  5. Due teste di classificazione (identity, orientation) fanno da task di
     pretraining supervisionato.
  6. Funzioni di verifica: accuracy per slot + controllo di permutazione
     valida.

Richiede:
  pip install magiccube networkx torch torch_geometric torch_scatter
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean

import magiccube
from magiccube.cube_base import Face, Color

# ----------------------------------------------------------------------
# 1) COSTANTI STRUTTURALI DEL CUBO (fisse, indipendenti dallo stato)
# ----------------------------------------------------------------------

SIZE = 3
N_COLORS = 6
COLOR_TO_IDX = {c: i for i, c in enumerate(Color)}

_template = magiccube.Cube(SIZE)

# 26 posizioni (slot) fisse, in un ordine canonico stabile
SLOT_COORDS = sorted(_template.get_all_pieces().keys())
COORD_TO_SLOT = {c: i for i, c in enumerate(SLOT_COORDS)}
N_SLOTS = len(SLOT_COORDS)
assert N_SLOTS == 26

HOME_COLORS = {i: _template.get_piece(c).get_piece_colors() for i, c in enumerate(SLOT_COORDS)}


def _applicable_axes(coord):
    return [ax for ax in range(3) if coord[ax] in (0, SIZE - 1)]


def _slot_type(coord):
    n = sum(1 for c in coord if c in (0, SIZE - 1))
    return {3: 'corner', 2: 'edge', 1: 'center'}[n]


SLOT_TYPE = [_slot_type(c) for c in SLOT_COORDS]


# ----------------------------------------------------------------------
# 2) cube2graph (topologia dell'utente, adattata)
#    - id fisso 0..53 per sticker, stesso ordine di iterazione di
#      cube.get_all_faces() (ordine dell'enum Face: L,R,D,U,B,F)
#    - la topologia (archi) e' calcolata UNA VOLTA su un cubo qualsiasi
#      (non dipende dallo stato, solo dalla geometria del cubo)
#    - i colori vengono estratti separatamente per ogni sample
# ----------------------------------------------------------------------

def _connect_face(G, face_ids):
    """Connessioni di griglia 3x3 dentro una faccia + scorciatoie angolo-centro
    (topologia originale di cube2graph)."""
    for row in range(3):
        for col in range(2):
            G.add_edge(face_ids[row * 3 + col], face_ids[row * 3 + col + 1])
    for col in range(3):
        for row in range(2):
            G.add_edge(face_ids[row * 3 + col], face_ids[(row + 1) * 3 + col])
    for idx in range(0, 3, 2):
        G.add_edge(face_ids[idx], face_ids[4])
        G.add_edge(face_ids[idx + 6], face_ids[4])


def _build_topology_and_id_map(cube):
    """
    Ricostruisce la topologia di cube2graph assegnando gli id (0..53) nello
    stesso ordine con cui cube2graph itera cube.get_all_faces(), e restituisce
    anche la coordinata 3D (slot fisico) di ogni id, cosi' da poter derivare
    sia gli archi PyG sia STICKER_TO_SLOT una volta sola.
    """
    G = nx.Graph()
    faces_ids, faces_coords = {}, {}
    next_id = 0
    for face in Face:
        face_indexes = cube._cube_face_indexes[face.value]
        coords_flat = [c for line in face_indexes for c in line]
        ids_flat = list(range(next_id, next_id + 9))
        faces_ids[face.name] = ids_flat
        faces_coords[face.name] = coords_flat
        G.add_nodes_from(ids_flat)
        next_id += 9

    for face_name, ids in faces_ids.items():
        _connect_face(G, ids)

    _horizontal = ['L', 'F', 'R', 'B']
    for i in range(len(_horizontal)):
        lx = faces_ids[_horizontal[i]]
        rx = faces_ids[_horizontal[(i + 1) % 4]]
        for j in range(0, 7, 3):
            G.add_edge(lx[j + 2], rx[j])

    G.add_edge(faces_ids['F'][0], faces_ids['U'][6])
    G.add_edge(faces_ids['F'][1], faces_ids['U'][7])
    G.add_edge(faces_ids['F'][2], faces_ids['U'][8])
    G.add_edge(faces_ids['F'][6], faces_ids['D'][0])
    G.add_edge(faces_ids['F'][7], faces_ids['D'][1])
    G.add_edge(faces_ids['F'][8], faces_ids['D'][2])
    G.add_edge(faces_ids['R'][0], faces_ids['U'][8])
    G.add_edge(faces_ids['R'][1], faces_ids['U'][5])
    G.add_edge(faces_ids['R'][2], faces_ids['U'][2])
    G.add_edge(faces_ids['R'][6], faces_ids['D'][2])
    G.add_edge(faces_ids['R'][7], faces_ids['D'][5])
    G.add_edge(faces_ids['R'][8], faces_ids['D'][8])
    G.add_edge(faces_ids['B'][0], faces_ids['U'][2])
    G.add_edge(faces_ids['B'][1], faces_ids['U'][1])
    G.add_edge(faces_ids['B'][2], faces_ids['U'][0])
    G.add_edge(faces_ids['B'][6], faces_ids['D'][8])
    G.add_edge(faces_ids['B'][7], faces_ids['D'][7])
    G.add_edge(faces_ids['B'][8], faces_ids['D'][6])
    G.add_edge(faces_ids['L'][0], faces_ids['U'][0])
    G.add_edge(faces_ids['L'][1], faces_ids['U'][3])
    G.add_edge(faces_ids['L'][2], faces_ids['U'][6])
    G.add_edge(faces_ids['L'][6], faces_ids['D'][6])
    G.add_edge(faces_ids['L'][7], faces_ids['D'][3])
    G.add_edge(faces_ids['L'][8], faces_ids['D'][0])

    id_to_coord = {}
    for face_name, ids in faces_ids.items():
        for nid, coord in zip(ids, faces_coords[face_name]):
            id_to_coord[nid] = coord

    return G, id_to_coord


_TOPOLOGY_GRAPH, _ID_TO_COORD = _build_topology_and_id_map(_template)
N_STICKERS = _TOPOLOGY_GRAPH.number_of_nodes()
assert N_STICKERS == 54

# edge_index fisso (PyG), calcolato una sola volta: from_networkx aggiunge
# gia' entrambe le direzioni per un grafo indiretto
FIXED_EDGE_INDEX = from_networkx(_TOPOLOGY_GRAPH).edge_index

# mappa sticker(0..53) -> slot fisico (0..25), derivata dalla stessa geometria
STICKER_TO_SLOT = torch.tensor(
    [COORD_TO_SLOT[_ID_TO_COORD[i]] for i in range(N_STICKERS)], dtype=torch.long
)


# ----------------------------------------------------------------------
# 3) GENERAZIONE CAMPIONI: scramble + estrazione colori (input, one-hot)
#    e ground truth (identity, orientation) via id(piece)
# ----------------------------------------------------------------------

def make_scrambled_sample(n_moves):
    cube = magiccube.Cube(SIZE)  # stato risolto, pezzi freschi
    id_to_origid = {id(cube.get_piece(c)): i for i, c in enumerate(SLOT_COORDS)}

    moves = cube.generate_random_moves(num_steps=n_moves, wide=False)
    cube.rotate(moves)

    # --- colori per i 54 sticker (input), stesso ordine di iterazione
    #     usato per costruire la topologia (Face enum, poi riga per riga) ---
    colors = torch.zeros(N_STICKERS, dtype=torch.long)
    next_id = 0
    for face in Face:
        flat_colors = cube.get_face_flat(face)  # 9 Color, ordine allineato a _cube_face_indexes
        for col in flat_colors:
            colors[next_id] = COLOR_TO_IDX[col]
            next_id += 1

    # --- ground truth identity/orientation per i 26 slot ---
    identity = torch.zeros(N_SLOTS, dtype=torch.long)
    orientation = torch.full((N_SLOTS,), -100, dtype=torch.long)  # -100 = ignora in CE loss
    for slot_idx, coord in enumerate(SLOT_COORDS):
        piece = cube.get_piece(coord)
        orig_id = id_to_origid[id(piece)]
        identity[slot_idx] = orig_id

        if SLOT_TYPE[slot_idx] == 'center':
            continue

        home = HOME_COLORS[orig_id]
        primary_axis = next(ax for ax in range(3) if home[ax] is not None)
        primary_color = home[primary_axis]
        cur = piece.get_piece_colors()
        axes = _applicable_axes(coord)
        ori_axis = next(ax for ax in axes if cur[ax] == primary_color)
        orientation[slot_idx] = axes.index(ori_axis)

    return colors, identity, orientation


def make_data_sample(n_moves):
    colors, identity, orientation = make_scrambled_sample(n_moves)
    x = F.one_hot(colors, num_classes=N_COLORS).float()
    data = Data(x=x, edge_index=FIXED_EDGE_INDEX)
    data.identity_y = identity
    data.orientation_y = orientation
    return data


# ----------------------------------------------------------------------
# 4) MODELLO: encoder GNN + pooling deterministico + teste di pretraining
# ----------------------------------------------------------------------

class StickerEncoder(nn.Module):
    def __init__(self, in_dim=N_COLORS, hidden=64, out_dim=64, n_layers=3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        dim = in_dim
        for i in range(n_layers):
            out = hidden if i < n_layers - 1 else out_dim
            self.convs.append(GATv2Conv(dim, out // heads, heads=heads))
            dim = out
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers - 1)] + [nn.LayerNorm(out_dim)]
        )

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
        return x  # [n_sticker_nodes_in_batch, out_dim]


def pool_stickers_to_pieces(sticker_emb, batch_vec, sticker_to_slot):
    """
    Pooling deterministico e NON learnable: media degli embedding degli
    sticker che appartengono allo stesso pezzo fisico (mappatura nota a
    priori, derivata dalla stessa geometria usata per costruire il grafo).
    """
    n_graphs = int(batch_vec.max().item()) + 1
    local_slot = sticker_to_slot.repeat(n_graphs).to(sticker_emb.device)
    global_slot = local_slot + batch_vec * N_SLOTS
    piece_emb = scatter_mean(sticker_emb, global_slot, dim=0, dim_size=n_graphs * N_SLOTS)
    piece_batch = torch.arange(n_graphs, device=sticker_emb.device).repeat_interleave(N_SLOTS)
    return piece_emb, piece_batch


class PieceHeads(nn.Module):
    """Teste di pretraining applicate a ciascuno dei 26 embedding di pezzo."""
    def __init__(self, in_dim=64, hidden=64):
        super().__init__()
        self.identity_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, N_SLOTS)
        )
        self.orientation_head = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 3)  # max 3 classi (corner)
        )

    def forward(self, piece_emb):
        return self.identity_head(piece_emb), self.orientation_head(piece_emb)


class RubikPretrainModel(nn.Module):
    def __init__(self, hidden=64, emb_dim=64):
        super().__init__()
        self.encoder = StickerEncoder(hidden=hidden, out_dim=emb_dim)
        self.heads = PieceHeads(in_dim=emb_dim, hidden=hidden)

    def forward(self, data):
        sticker_emb = self.encoder(data.x, data.edge_index)
        piece_emb, piece_batch = pool_stickers_to_pieces(sticker_emb, data.batch, STICKER_TO_SLOT)
        id_logits, ori_logits = self.heads(piece_emb)
        return piece_emb, id_logits, ori_logits, piece_batch


# ----------------------------------------------------------------------
# 5) TRAINING LOOP (curriculum: scramble corti -> lunghi)
# ----------------------------------------------------------------------

def train(n_steps=2000, batch_size=64, lr=1e-3, device='cpu', curriculum=True):
    model = RubikPretrainModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, n_steps + 1):
        n_moves = min(2 + step // 100, 25) if curriculum else 25

        batch_list = [make_data_sample(n_moves) for _ in range(batch_size)]
        batch = Batch.from_data_list(batch_list).to(device)

        _, id_logits, ori_logits, _ = model(batch)
        identity_y = batch.identity_y
        orientation_y = batch.orientation_y

        loss_id = F.cross_entropy(id_logits, identity_y)
        loss_ori = F.cross_entropy(ori_logits, orientation_y, ignore_index=-100)
        loss = loss_id + loss_ori

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                id_acc = (id_logits.argmax(-1) == identity_y).float().mean().item()
                mask = orientation_y != -100
                ori_acc = (ori_logits.argmax(-1)[mask] == orientation_y[mask]).float().mean().item()
            print(f"step {step:5d} | n_moves={n_moves:2d} | loss={loss.item():.4f} "
                  f"| identity_acc={id_acc:.3f} | orientation_acc={ori_acc:.3f}")

    return model


# ----------------------------------------------------------------------
# 6) VERIFICA: il grafo di output a 26 nodi rappresenta correttamente i pezzi?
# ----------------------------------------------------------------------

@torch.no_grad()
def verify_piece_graph(model, n_samples=500, n_moves=25, device='cpu'):
    model.eval()
    data_list = [make_data_sample(n_moves) for _ in range(n_samples)]
    batch = Batch.from_data_list(data_list).to(device)

    _, id_logits, ori_logits, _ = model(batch)
    id_pred = id_logits.argmax(-1).view(n_samples, N_SLOTS)
    ori_pred = ori_logits.argmax(-1).view(n_samples, N_SLOTS)
    id_true = batch.identity_y.view(n_samples, N_SLOTS)
    ori_true = batch.orientation_y.view(n_samples, N_SLOTS)

    identity_acc = (id_pred == id_true).float().mean().item()
    ori_mask = ori_true != -100
    orientation_acc = (ori_pred[ori_mask] == ori_true[ori_mask]).float().mean().item()
    exact_mask = (id_pred == id_true) & ((ori_pred == ori_true) | ~ori_mask)
    exact_match_acc = exact_mask.float().mean().item()

    corner_slots = [i for i, t in enumerate(SLOT_TYPE) if t == 'corner']
    edge_slots = [i for i, t in enumerate(SLOT_TYPE) if t == 'edge']
    valid_perm_count = 0
    for b in range(n_samples):
        corner_ids = id_pred[b, corner_slots].tolist()
        edge_ids = id_pred[b, edge_slots].tolist()
        if len(set(corner_ids)) == len(corner_ids) and len(set(edge_ids)) == len(edge_ids):
            valid_perm_count += 1
    permutation_validity = valid_perm_count / n_samples

    print("---- Verifica del grafo a 26 nodi (piece-level) ----")
    print(f"identity_accuracy      (per-slot): {identity_acc:.4f}")
    print(f"orientation_accuracy   (per-slot): {orientation_acc:.4f}")
    print(f"exact_match_accuracy   (per-slot): {exact_match_acc:.4f}")
    print(f"permutation_validity (per-cubo)  : {permutation_validity:.4f}")
    print("  (permutation_validity=1.0 => nessun cubo ha pezzi duplicati "
          "nelle predizioni: la rappresentazione e' fisicamente coerente)")

    return {
        "identity_accuracy": identity_acc,
        "orientation_accuracy": orientation_acc,
        "exact_match_accuracy": exact_match_acc,
        "permutation_validity": permutation_validity,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    model = train(n_steps=2000, batch_size=64, curriculum=True)
    verify_piece_graph(model, n_samples=500, n_moves=25)

    torch.save(model.encoder.state_dict(), "sticker_encoder_pretrained.pt")
    print("Encoder salvato in sticker_encoder_pretrained.pt")
