from magiccube import Cube


def get_other_orientations(piece_orientation: list):
    piece_type = len(piece_orientation)
    other_orientations = [piece_orientation.copy()]
    if piece_type == 3:
        for _ in range(2):
            piece_orientation.append(piece_orientation.pop(0))
            other_orientations.append(piece_orientation.copy())
    if piece_type == 2:
        other_orientations.append([piece_orientation[1], piece_orientation[0]])

    return other_orientations


# extract position and orientations from a cube
def cube2pieces(cube: Cube) -> dict:
    position_label = {}
    orientations_label = {}

    for index, (coord, piece) in enumerate(cube.get_all_pieces().items()):
        colors = [c.value for c in piece.get_piece_colors() if c is not None]

        orientations_label[index] = get_other_orientations(colors)
        position_label[index] = colors

    return {
        "position": position_label,
        "orientation": orientations_label
    }


# returns the label associated by the state of the cube
def mapper(cube: Cube) -> list:
    ref_orientation = cube2pieces(Cube())["orientation"]
    cur_orientation = cube2pieces(cube)["orientation"]

    label = []

    for idx in ref_orientation:
        ref_value = ref_orientation[idx]
        cur_value = cur_orientation[idx]

        if ref_value[0] == cur_value[0]:
            label.append(1.0)
        elif cur_value[0] in ref_value:
            label.append(0.0)
        else:
            label.append(-1.0)

    return label