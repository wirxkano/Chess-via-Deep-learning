import os
import pickle
import torch
import torch.nn as nn
from chess import pgn
from torch.utils.data import DataLoader
from Utils.auxiliary import *
from DeepLearning.Dataset.ChessDataset import ChessDataset
from tqdm import tqdm

class Trainer:
    def __init__(self, network_class, pgn_file_path, limit_files, learning_rate, batch_size, epochs, snapshot_path=None, save_path=None):
        self.network_class = network_class
        self.pgn_file_path = pgn_file_path
        self.limit_files = limit_files
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.cur_epoch = 0
        
        self.snapshot_path = snapshot_path
        self.save_path = save_path
        
        self.loss = 1000000
        if self.snapshot_path:
            self._load_snapshot(self.snapshot_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X, self.y, self.num_classes, self.move_to_int = self.get_Xy()
        
        self.train_loader = self._build_dataloader(True)
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
    def train(self):
        for epoch in tqdm(range(self.cur_epoch, self.epochs)):
            self.model.train()
            loss = self.train_step()
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss / len(self.train_loader):.4f}')
            
            # Save best
            if loss < self.loss:
                self._save_snapshot(epoch, loss, "best")
                self.loss = loss
            # Save period
            self.save_path(epoch, loss, "last")
         
        with open("./checkpoints/heavy_move_to_int", "wb") as file:
            pickle.dump(self.move_to_int, file)
    
    def train_step(self):
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            running_loss += loss.item()
            
        return running_loss
      
    def _load_snapshot(self, path):
        if not path:
            raise ValueError("Invalid snapshot path")
        
        snapshot = torch.load(path, weights_only=True)
        self.model.load_state_dict(snapshot["model"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.cur_epoch = snapshot["epoch"]
        self.loss = snapshot["loss"]
        
    def _save_snapshot(self, epoch, loss, name):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "loss": loss
        }, os.path.join(self.save_path, f"{name}.pth"))
        
        print(f"Save model at epoch {epoch+1}\n")
         
    def _build_dataloader(self, train=True):
        dataset = ChessDataset(self.X, self.y)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4, shuffle=True if train else False)
      
    def _build_model(self):
        return self.network_class(num_classes=self.num_classes).to(self.device)
      
    def _build_criterion(self):
        return nn.CrossEntropyLoss()
      
    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
      
    def _build_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
      
    def load_pgn(self, file_path):
        games = []
        with open(file_path, 'r') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        return games
      
    def load_data(self):
        files = [file for file in os.listdir(self.pgn_file_path) if file.endswith(".pgn")]
        LIMIT_OF_FILES = min(len(files), self.limit_files)
        games = []
        i = 1
        for file in tqdm(files):
            games.extend(self.load_pgn(f"{self.pgn_file_path}/{file}"))
            if i >= LIMIT_OF_FILES:
                break
            i += 1
        return games
      
    def get_Xy(self):
        X, y = preprocess(self.load_data())
        y, move_to_int = encode_moves(y)
        num_classes = len(move_to_int)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y, num_classes, move_to_int
      
    