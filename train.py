import torch
import torch.nn as nn
from models.point_encoder import PointEncoder
from models.transformer import DeepMeshTransformer
import argparse

def train():
    parser = argparse.ArgumentParser(description="Treinamento do DeepMesh Reproduction")
    parser.add_argument('--epochs', type=int, default=10, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=2, help='Tamanho do lote')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Treinando na device: {device}")
    
    # Instanciando modelos
    encoder = PointEncoder().to(device)
    transformer = DeepMeshTransformer().to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(transformer.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("Iniciando loop de treinamento...")
    for epoch in range(1, args.epochs + 1):
        # Gerando dados fictícios para demonstração do Pipeline
        # (B, N, 6)
        pc_mock = torch.rand(args.batch_size, 4096, 6).to(device)
        # Sequência de blocos auto-regressivos: (B, max_seq_len)
        targets_mock = torch.randint(0, 50254, (args.batch_size, 4500)).to(device)
        
        optimizer.zero_grad()
        
        # 1. Point Encoder: Extração de Features (B, N, 256)
        pc_embeds = encoder(pc_mock)
        
        # 2. Transformer: Predição dos logits do vocabulário
        logits = transformer(targets_mock, pc_embeds)
        
        # Shift para treinamento autoregressivo predizendo o próximo token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_mock[..., 1:].contiguous()
        
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()
