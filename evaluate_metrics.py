import argparse
import open3d as o3d
import numpy as np
from utils.metrics import compute_chamfer_distance, compute_hausdorff_distance

def main():
    parser = argparse.ArgumentParser(description="Avaliação de Métricas (Chamfer e Hausdorff)")
    parser.add_argument('--gt', type=str, required=True, help="Caminho para o OBJ original (Ground Truth)")
    parser.add_argument('--pred', type=str, required=True, help="Caminho para o OBJ reconstruído (ex: gerado pelo nosso código, MeshAnythingv2 ou BPT)")
    parser.add_argument('--num_points', type=int, default=10000, help="Pontos amostrados para a métrica")
    args = parser.parse_args()

    import trimesh
    print(f"Carregando Ground Truth: {args.gt}")
    gt_mesh = trimesh.load(args.gt, force='mesh')
    
    print(f"Carregando Predição: {args.pred}")
    pred_mesh = trimesh.load(args.pred, force='mesh')

    # Amostrar nuvens de pontos densas para calcular a distância
    gt_pc, _ = trimesh.sample.sample_surface(gt_mesh, args.num_points)
    pred_pc, _ = trimesh.sample.sample_surface(pred_mesh, args.num_points)
    
    # Normalizar ambas as nuvens (opcional, mas recomendado para justificar C.Dist e H.Dist na mesma escala)
    gt_centroid = np.mean(gt_pc, axis=0)
    gt_pc -= gt_centroid
    gt_max = np.max(np.sqrt(np.sum(gt_pc**2, axis=1)))
    gt_pc /= gt_max
    
    pred_centroid = np.mean(pred_pc, axis=0)
    pred_pc -= pred_centroid
    pred_max = np.max(np.sqrt(np.sum(pred_pc**2, axis=1)))
    pred_pc /= pred_max
    
    # Calcular Métricas
    cdist = compute_chamfer_distance(gt_pc, pred_pc)
    hdist = compute_hausdorff_distance(gt_pc, pred_pc)
    
    print("\n" + "="*40)
    print(" RESULTADOS DAS MÉTRICAS PARA A TABELA ")
    print("="*40)
    print(f"Modelo Avaliado: {args.pred}")
    print(f"Chamfer Distance (C.Dist.)  : {cdist:.6f}")
    print(f"Hausdorff Distance (H.Dist.): {hdist:.6f}")
    print("="*40)

if __name__ == '__main__':
    main()
