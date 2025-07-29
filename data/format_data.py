from magiccube import Cube
import json
import time


data = json.load(open("data.json", "r"))

for element in data.keys():
    cube = Cube()

    instance = data[element]
    scramble:str = instance["scramble"]
    solution:str = instance["solution"]

    cube.rotate(scramble)

    print(cube)
    time.sleep(1)
    
    formatted_solution = {}
    
    for rotation in solution.split():
        cube.rotate(rotation)
        print(cube)