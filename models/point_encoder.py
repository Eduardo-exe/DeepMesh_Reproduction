import torch
import torch.nn as nn
import torch.nn.functional as F

class PointEncoder(nn.Module):
    """
    Point Cloud Encoder inspirado na arquitetura PointNet.
    Extrai features locais de cada ponto e combina com features globais,
    garantindo que informações geométricas da malha sejam representadas.
    
    Entrada: (B, N, 6) onde N é o número de pontos e 6 = (x, y, z, nx, ny, nz)
    Saída: (B, N, 256)
    """
    def __init__(self, in_channels=6, out_channels=256):
        super(PointEncoder, self).__init__()
        
        # Primeira extração local
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # MLPs para fusão de features globais e locais
        self.conv4 = nn.Conv1d(256 + 64, 256, 1)
        self.bn4 = nn.BatchNorm1d(256)
        
        self.out_dim = out_channels
        
    def forward(self, x):
        """
        Passagem forward do PointEncoder.
        x: Tensor de shape (B, N, 6)
        Retorna: Tensor de shape (B, N, 256)
        """
        # PyTorch Conv1d espera a dimensão dos canais no meio: (B, C, N)
        x = x.transpose(1, 2) 
        N = x.size(-1)
        
        # Extração de features locais
        local_feat = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        
        # Progressão para dimensão maior
        net = F.relu(self.bn2(self.conv2(local_feat))) # (B, 128, N)
        net = self.bn3(self.conv3(net)) # (B, 256, N)
        
        # Pooling global (Max Pooling ao longo da dimensão N)
        global_feat = torch.max(net, 2, keepdim=True)[0] # (B, 256, 1)
        
        # Expande as features globais para concatebar com cada ponto
        global_feat_expanded = global_feat.repeat(1, 1, N) # (B, 256, N)
        
        # Concatenação: Combina feature global da geometria + features locais (B, 320, N)
        combined = torch.cat([local_feat, global_feat_expanded], dim=1) 
        
        # Processamento final para reduzir à dimensão de saída desejada
        out = F.relu(self.bn4(self.conv4(combined))) # (B, 256, N)
        
        # Retorna na orientação (B, N, C) para uso direto no Transformer
        return out.transpose(1, 2)
