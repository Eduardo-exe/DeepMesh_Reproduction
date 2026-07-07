import open3d as o3d
import numpy as np
import trimesh

def reconstruct_mesh_ball_pivoting(vertices):
    """
    Reconstrução utilizando algoritmo Ball Pivoting via Open3D.
    Ideal para nuvens de pontos densas.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2])
    )
    
    return bpa_mesh

def reconstruct_mesh_poisson(vertices, depth=8):
    """
    Reconstrução utilizando Poisson Surface Reconstruction.
    Melhor tolerância a ruído, gera malhas contínuas e herméticas (watertight).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    
    # Remover malhas geradas falsamente em regiões de baixa densidade
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Corrigir normais para renderizar perfeitamente no MeshLab
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # Aplicar gradiente idêntico ao do Paper original (Roxo em baixo -> Amarelinho/Branco em cima)
    verts = np.asarray(mesh.vertices)
    if len(verts) > 0:
        y_vals = verts[:, 1]
        y_min, y_max = np.min(y_vals), np.max(y_vals)
        y_norm = (y_vals - y_min) / (y_max - y_min + 1e-6)
        
        color_bottom = np.array([0.6, 0.2, 0.8]) # Roxo
        color_top = np.array([1.0, 0.98, 0.90])   # Branco amarelado
        
        colors = (1 - y_norm[:, None]) * color_bottom + y_norm[:, None] * color_top
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    return mesh
