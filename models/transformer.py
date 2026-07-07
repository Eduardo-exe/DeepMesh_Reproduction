import torch
import torch.nn as nn
from .attention import precompute_freqs_cis
from .hourglass import HourglassTransformer

class DeepMeshTransformer(nn.Module):
    """
    Modelo Transformer Auto-Regressivo condicionado por nuvens de pontos.
    Utiliza uma arquitetura Hourglass para processar longas sequências de tokens
    geométricos (patch -> block -> offset).
    """
    def __init__(self, vocab_size=50254, d_model=256, n_heads=8, max_seq_len=4500, context_dim=256):
        super().__init__()
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Fatores da arquitetura Hourglass: [pre_layers, down, pre_inner_layers, inner_down...]
        factors = [2, 1, 2, 3, 2, 3] 
        self.transformer = HourglassTransformer(None, vocab_size, n_heads, d_model, factors, context_dim)
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len * 2)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        
    def forward(self, idx, pc_embeds=None):
        """
        idx: tokens de entrada (B, T)
        pc_embeds: embeddings da nuvem de pontos (B, N, context_dim)
        """
        B, T = idx.size()
        
        # Máscara Causal: proíbe "espiar" tokens futuros
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        
        x = self.token_emb(idx)
        
        freqs_cis = self.freqs_cis[:T].to(idx.device)
        
        out = self.transformer(x, mask=mask, freqs_cis=freqs_cis, context=pc_embeds)
        
        out = self.norm(out)
        logits = self.lm_head(out)
        
        return logits
