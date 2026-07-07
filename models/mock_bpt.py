import open3d as o3d
import numpy as np
from .mesh_reconstruction import reconstruct_mesh_poisson

def simulate_bpt(vertices, voxel_size=0.03):
    """
    Simula os artefatos visuais do BPT (Blocked and Patchified Tokenization).
    O BPT divide a forma em blocos/patches, o que gera desconexões e perda de resolução fina.
    Simulamos isso com Voxel Downsampling para mesclar detalhes em "blocos" isolados,
    e um ruído Gaussiano nas pontas para simular quebra de blocos.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Downsampling em blocos/voxels (Simulando "Patchified Blocks")
    down_pcd = pcd.voxel_down_sample(voxel_size)
    v_down = np.asarray(down_pcd.points)
    
    # Ruído para simular quebra nas bordas dos patches
    noise = np.random.normal(0, voxel_size * 0.15, v_down.shape)
    v_noisy = v_down + noise
    
    mesh = reconstruct_mesh_poisson(v_noisy, depth=8)
    return mesh, v_noisy
