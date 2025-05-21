import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

import pandas as pd
import pickle
from Utils.auxiliary import *
from tqdm import tqdm

all_moves = set()
df= pd.read_csv("DeepLearning/data/lichess-08-2014.csv", nrows=200000)

for pgn in tqdm(df["PGN"]):
    _, y = preprocess(pgn)
    all_moves.update(y)

move_to_int = {move: idx for idx, move in enumerate(sorted(all_moves))}
print(len(move_to_int))

with open("DeepLearning/checkpoints/move_to_int_200k.pkl", "wb") as f:
    pickle.dump(move_to_int, f)
    print("Saved!")