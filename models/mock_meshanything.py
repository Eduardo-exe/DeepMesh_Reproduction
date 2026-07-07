import open3d as o3d
import numpy as np
from .mesh_reconstruction import reconstruct_mesh_poisson

def simulate_meshanything(vertices, grid_resolution=128):
    """
    Simula os artefatos visuais do MeshAnythingV2.
    O MeshAnything usa Adjacent Mesh Tokenization, discretizando o espaço contínuo em tokens.
    Simulamos isso aplicando quantização agressiva nas coordenadas antes de reconstruir.
    """
    # Escalar para [0, grid_resolution]
    v_min, v_max = np.min(vertices, axis=0), np.max(vertices, axis=0)
    v_scaled = (vertices - v_min) / (v_max - v_min + 1e-6) * grid_resolution
    
    # Quantizar (Arredondar para o token mais próximo)
    v_quantized = np.round(v_scaled)
    
    # Retornar à escala original
    v_reconstructed = (v_quantized / grid_resolution) * (v_max - v_min) + v_min
    
    # Adicionar ruído numérico ínfimo para evitar normais zeradas no Poisson
    v_reconstructed += np.random.normal(0, 1e-5, v_reconstructed.shape)
    
    # Reconstruir usando nossa pipeline
    mesh = reconstruct_mesh_poisson(v_reconstructed, depth=8)
    return mesh, v_reconstructed
