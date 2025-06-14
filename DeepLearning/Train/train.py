import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from Trainer import Trainer
from DeepLearning.Model.ChessModel import ChessResNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mapping_data_path', type=str, required=True)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default='DeepLearning/checkpoints')

    args = parser.parse_args()

    trainer = Trainer(
        ChessResNet,
        args.csv_path,
        args.lr,
        args.batch,
        args.epochs,
        args.mapping_data_path,
        args.snapshot,
        args.checkpoint
    )
    
    trainer.train()
