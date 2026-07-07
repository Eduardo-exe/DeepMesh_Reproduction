import argparse
import open3d as o3d
import numpy as np
import os

def process_mesh_to_point_cloud(mesh_path, num_points=4096):
    """
    Carrega o OBJ usando Open3D, calcula as normais das faces,
    amostra pontos uniformemente e normaliza os pontos.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # Amostrar nuvem de pontos uniformemente
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Normalizar coordenadas para [-1, 1] no centro geométrico
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= max_dist
    
    return points, normals

def main():
    parser = argparse.ArgumentParser(description="Conversor de Dataset ShapeNet -> NPZ (Point Cloud)")
    parser.add_argument('--input', type=str, required=True, help='Caminho para arquivo .obj de entrada')
    parser.add_argument('--output', type=str, required=True, help='Caminho para arquivo .npz de saída')
    args = parser.parse_args()
    
    print(f"Processando {args.input}...")
    points, normals = process_mesh_to_point_cloud(args.input)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, points=points, normals=normals)
    print(f"Dataset processado salvo com sucesso em {args.output}")

if __name__ == '__main__':
    main()
