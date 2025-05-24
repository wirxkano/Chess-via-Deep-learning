import torch
from Utils.auxiliary import *
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self, df_chunk, move_to_int):
        self.df = df_chunk
        self.samples = []

        for pgn in self.df["PGN"]:
            try:
                X, y = preprocess(pgn)
                self.samples.extend(zip(X, y))
            except:
                continue
            
        self.labels = encode_moves([move for _, move in self.samples], move_to_int)
        self.samples = [(board_tensor, label) for (board_tensor, _), label in zip(self.samples, self.labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board_tensor, label = self.samples[idx]
        return torch.tensor(board_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    