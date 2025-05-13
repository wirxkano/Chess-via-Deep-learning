import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from Trainer import Trainer
from DeepLearning.Model.ChessModel import ChessModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--limit_files', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints')

    args = parser.parse_args()

    trainer = Trainer(
        ChessModel,
        args.pgn_path,
        args.limit_files,
        args.lr,
        args.batch,
        args.epochs,
        args.snapshot,
        args.checkpoint
    )
    
    trainer.train()
