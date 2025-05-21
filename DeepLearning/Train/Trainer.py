import os
import pickle
import gc
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from Utils.auxiliary import *
from DeepLearning.Dataset.ChessDataset import ChessDataset
from tqdm import tqdm

class Trainer:
    def __init__(self, network_class, csv_file_path, learning_rate, batch_size, epochs, mapping_data_path=None, snapshot_path=None, save_path=None):
        self.network_class = network_class
        self.csv_file_path = csv_file_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.cur_epoch = 0
        
        self.snapshot_path = snapshot_path
        self.save_path = save_path
        
        self.loss = 1000000
        
        with open(mapping_data_path, "rb") as f:
            self.move_to_int = pickle.load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = len(self.move_to_int)
        
        # self.train_loader = self._build_dataloader(df_chunk)
        
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        if self.snapshot_path:
            self._load_snapshot(self.snapshot_path)
        
    def train(self):
        df = pd.read_csv(self.csv_file_path, usecols=["PGN"], nrows=300000)
        total_games = len(df)
        chunk_size = 30000
        
        for epoch in tqdm(range(self.cur_epoch, self.epochs)):
            loss = 0.0
            num_batches = 0
            for start in range(0, total_games, chunk_size):
                end = min(start + chunk_size, total_games)
                print(f"\nLoading games {start} to {end}...")
                df_chunk = df.iloc[start:end]
                self.train_loader = self._build_dataloader(df_chunk)
                
                self.model.train()
                local_loss, local_batches = self.train_step()
                loss += local_loss
                num_batches += local_batches
                
                del self.train_loader
                del df_chunk
                gc.collect()
               
            avg_loss = loss / num_batches
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}')
            
            # Save period
            self._save_snapshot(epoch, avg_loss, f"last_{epoch+1}")
            # Save best
            if avg_loss < self.loss:
                self._save_snapshot(epoch, avg_loss, "best")
                self.loss = avg_loss
    
    def train_step(self):
        running_loss = 0.0
        num_batches = 0
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
            num_batches += 1
            
        return running_loss, num_batches
    
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
            "epoch": epoch+1,
            "loss": loss
        }, os.path.join(self.save_path, f"{name}.pth"))
        
        print(f"Save model at epoch {epoch+1}\n")
    
    def _build_dataloader(self, df_chunk):
        dataset = ChessDataset(df_chunk, self.move_to_int)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
    
    def _build_model(self):
        return self.network_class(num_classes=self.num_classes).to(self.device)
    
    def _build_criterion(self):
        return nn.CrossEntropyLoss()
    
    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _build_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)
    