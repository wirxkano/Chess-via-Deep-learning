from configs import *
from Minimax.Minimax import Minimax
from Random.RandomAgent import RandomAgent
from DeepLearning.predict import DeepLearningAgent

class Engine:
  def __init__(self, method, depth=0):
    self.method = method
    self.depth = depth
  
  def move(self, board, color):
    if type(self.method) is Minimax:
      move, _ = self.method.minimax(board, self.depth, -CHECKMATE, CHECKMATE, color)
      return move
    
    elif type(self.method) is RandomAgent:
      return self.method.get_move(board, color)
    
    elif type(self.method) is DeepLearningAgent:
      return self.method.predict_move(board, color)
      