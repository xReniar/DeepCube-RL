import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import Cube
import random
import multiprocessing


moves = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"]

def scramble_and_get_state(scramble):
    cube = Cube()
    cube.rotate(scramble)
    return cube

if __name__ == "__main__":
    dataset = set()
    for i in range(0, 10000):
        dataset.add(' '.join(random.choices(moves, k=20)))

    '''
    results = []
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(scramble_and_get_state, list(dataset))

    for result in results:
        print(result.graph_state()[0])
    '''

    dataset = list(map(lambda x: f"{x}\n", list(dataset)))
    with open("scrambles.txt", "w") as f:
        f.writelines(dataset)