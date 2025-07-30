from magiccube import Cube
import json
import time

face_names = ['U', 'R', 'F', 'D', 'L', 'B']


def check_cross(cube: Cube) -> bool:
    global face_names
    facelets = cube.get_kociemba_facelet_positions()

    faces = {face: facelets[i*9:(i+1)*9] for i, face in enumerate(face_names)}
    
    pieces = [
        faces["F"][4] == faces["F"][7] and faces["D"][1] == faces["D"][4],
        faces["R"][4] == faces["R"][7] and faces["D"][5] == faces["D"][4],
        faces["B"][4] == faces["B"][7] and faces["D"][7] == faces["D"][4],
        faces["L"][4] == faces["L"][7] and faces["D"][3] == faces["D"][4]
    ]

    return all(pieces)

def first_layer(cube: Cube) -> bool:
    global face_names
    facelets = cube.get_kociemba_facelet_positions()

    faces = {face: facelets[i*9:(i+1)*9] for i, face in enumerate(face_names)}

    pieces = [
        faces["F"][4] == faces["F"][8] and faces["D"][1] == faces["D"][2],
        faces["R"][4] == faces["R"][8] and faces["D"][5] == faces["D"][8],
        faces["B"][4] == faces["B"][8] and faces["D"][7] == faces["D"][6],
        faces["L"][4] == faces["L"][8] and faces["D"][3] == faces["D"][0]
    ]

    return all(pieces)

def second_layer(cube: Cube) -> bool:
    global face_names
    facelets = cube.get_kociemba_facelet_positions()

    faces = {face: facelets[i*9:(i+1)*9] for i, face in enumerate(face_names)}

    pieces = [
        faces["F"][4] == faces["F"][5] and faces["R"][4] == faces["R"][3],
        faces["R"][4] == faces["R"][5] and faces["B"][4] == faces["B"][3],
        faces["B"][4] == faces["B"][5] and faces["L"][4] == faces["L"][3],
        faces["L"][4] == faces["L"][5] and faces["F"][4] == faces["F"][3]
    ]

    return all(pieces)

def final_layer(cube: Cube) -> bool:
    pass


data = json.load(open("data.json", "r"))
status = [
    check_cross,
    first_layer,
    second_layer,
    final_layer        
]

def format_data():
    new_data = {}

    for element in data.keys():
        cube = Cube()

        instance = data[element]
        scramble:str = instance["scramble"]
        solution:str = instance["solution"]

        cube.rotate(scramble)

        steps = { 0: [], 1: [], 2: [], 3: [] }
        curr_solution = { 0: [], 1: [], 2: [], 3: [] }
        index = 0
        
        for rotation in solution.split():
            cube.rotate(rotation)
            curr_solution[index].append(rotation)

            if status[index](cube) == True:
                steps[index] = " ".join(curr_solution[index])
                index += 1

            if index == 3:
                break
                
        new_data[element] = dict(
            scramble = scramble,
            solution = dict(
                cross = steps[0],
                first_layer = steps[1],
                second_layer = steps[2]
            )
        )

    with open("dataset.json", "w") as f:
        json.dump(new_data, f, indent=4)


if __name__ == "__main__":
    format_data()