import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Pré-computa frequências complexas para Rotary Position Embeddings (RoPE).
    Não depende de kernels CUDA nativos.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Aplica RoPE. 
    Transforma vetores reais em complexos para rotação no plano e reverte para reais.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # Ajusta (B, Seq, Heads, Dim)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    """
    Módulo nativo de Multi-Head Self-Attention sem FlashAttention,
    ideal para compatibilidade em GPUs modestas, suportando RoPE.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None, freqs_cis=None):
        B, seq_len, _ = x.shape
        
        q = self.wq(x).view(B, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, seq_len, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, seq_len, self.n_heads, self.head_dim)
        
        if freqs_cis is not None:
            freqs_cis_cur = freqs_cis[:seq_len].to(x.device)
            q, k = apply_rotary_emb(q, k, freqs_cis_cur)
            
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)
        
        return self.wo(out)

class CrossAttention(nn.Module):
    """
    Cross-Attention para conectar o modelo autoregressivo aos embeddings de condição
    (ex: Embeddings Point Cloud).
    """
    def __init__(self, d_model, context_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(context_dim, d_model, bias=False)
        self.wv = nn.Linear(context_dim, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, context):
        B, N, _ = x.shape
        _, M, _ = context.shape
        
        q = self.wq(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(context).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(context).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.wo(out)


class FeedForward(nn.Module):
    """Implementação baseada em SwiGLU utilizada em LLMs modernos."""
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Bloco completo do Transformer contendo Self, Cross, FFN e Norms"""
    def __init__(self, d_model, n_heads, context_dim=None):
        super().__init__()
        self.attention = Attention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_model * 4)
        
        # Utilizando RMSNorm/LayerNorm do Pytorch Puro
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.cross_attention = CrossAttention(d_model, context_dim, n_heads)
            self.norm_cross = nn.LayerNorm(d_model)
            
    def forward(self, x, mask=None, freqs_cis=None, context=None):
        # 1. Self-Attention (com residual)
        h = x + self.attention(self.norm1(x), mask, freqs_cis)
        
        # 2. Cross-Attention
        if self.has_cross_attn and context is not None:
            h = h + self.cross_attention(self.norm_cross(h), context)
            
        # 3. Feed-Forward
        out = h + self.ffn(self.norm2(h))
        
        return out
