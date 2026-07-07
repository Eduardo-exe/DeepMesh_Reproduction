import numpy as np
from scipy.spatial import cKDTree
import time
import torch

def compute_chamfer_distance(pc1, pc2):
    """
    Distância de Chamfer L2 entre duas nuvens de pontos usando cKDTree.
    Medida simétrica.
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    dist1, _ = tree1.query(pc2, k=1)
    dist2, _ = tree2.query(pc1, k=1)
    
    chamfer_dist = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer_dist

def compute_hausdorff_distance(pc1, pc2):
    """
    Distância de Hausdorff direcional máxima.
    Sensível a outliers.
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    dist1, _ = tree1.query(pc2, k=1)
    dist2, _ = tree2.query(pc1, k=1)
    
    return max(np.max(dist1), np.max(dist2))

def compute_f_score(pc1, pc2, threshold=0.01):
    """
    F-Score em um threshold dado (ex: 1%).
    """
    tree1 = cKDTree(pc1)
    tree2 = cKDTree(pc2)
    
    dist1, _ = tree1.query(pc2, k=1)
    dist2, _ = tree2.query(pc1, k=1)
    
    precision = np.mean(dist1 < threshold)
    recall = np.mean(dist2 < threshold)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        return {"allocated_mb": allocated, "reserved_mb": reserved}
    return {"allocated_mb": 0, "reserved_mb": 0}

class Timer:
    def __init__(self):
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def end(self):
        return time.time() - self.start_time
