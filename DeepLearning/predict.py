import sys
import os
import pickle
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from Utils.auxiliary import *
from DeepLearning.Model.ChessModel import ChessResNet, ChessModel

class DeepLearningAgent:
    def __init__(self):
        with open("DeepLearning/checkpoints/move_to_int_300k.pkl", "rb") as file:
            move_to_int = pickle.load(file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessResNet(num_classes=len(move_to_int))
        ckpt = torch.load("DeepLearning/checkpoints/z4_0.pt", map_location=torch.device('cpu'))
        # self.model.load_state_dict(ckpt["model"])
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.int_to_move = {v: k for k, v in move_to_int.items()}
    
    def prepare_input(self, board):
        matrix = board_to_matrix(board)
        X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
        return X_tensor
      
    def predict_move(self, board, color):
        X_tensor = self.prepare_input(board).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
        
        logits = logits.squeeze(0)
        
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
        legal_moves = list(board.get_all_moves(color))
        sorted_indices = np.argsort(probabilities)[::-1]
        
        for move_index in sorted_indices:
            move = self.int_to_move[move_index]
            move = label_to_move(move)
            if move in legal_moves:
                return move
        
        return None