import torch
import torch.nn as nn
from einops import rearrange
from .attention import TransformerBlock

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        B, seq_len, D = x.shape
        remainder = seq_len % self.shorten_factor
        if remainder > 0:
            pad = self.shorten_factor - remainder
            x = torch.cat([x, torch.zeros(B, pad, D, device=x.device, dtype=x.dtype)], dim=1)
        
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x, orig_seq_len):
        x = self.proj(x)
        x = rearrange(x, 'b n (s d) -> b (n s) d', s=self.shorten_factor)
        return x[:, :orig_seq_len, :]

class HourglassTransformer(nn.Module):
    """
    Estrutura Hourglass original do artigo. Agrupa sequencias locais reduzindo
    o custo computacional.
    """
    def __init__(self, config, vocab_size, n_heads, n_embedding, factors, context_dim=256):
        super().__init__()
        self.n_embedding = n_embedding
        self.factors = factors
        self.n_layers = factors[0]
        
        self.pre_layer = nn.ModuleList([TransformerBlock(n_embedding, n_heads, context_dim) for _ in range(self.n_layers)])
        
        if len(self.factors) == 2:
            self.hourglass = None
        else:
            self.k = factors[3]
            self.downsample = LinearDownsample(n_embedding, self.k)
            self.upsample = LinearUpsample(n_embedding, self.k)
            
            self.hourglass = HourglassTransformer(
                config, vocab_size, n_heads, n_embedding, factors[2:], context_dim
            )
            self.post_layer = nn.ModuleList([TransformerBlock(n_embedding, n_heads, context_dim) for _ in range(self.n_layers)])
            
    def forward(self, x, mask=None, freqs_cis=None, context=None):
        orig_seq_len = x.size(1)
        
        for layer in self.pre_layer:
            x = layer(x, mask=mask, freqs_cis=freqs_cis, context=context)
            
        if self.hourglass is not None:
            x_residual = x.clone()
            
            x_down = self.downsample(x)
            
            # Forward interno (downsampled)
            x_down = self.hourglass(x_down, mask=None, freqs_cis=freqs_cis, context=context)
            
            x_up = self.upsample(x_down, orig_seq_len)
            
            x = x_up + x_residual
            
            for layer in self.post_layer:
                x = layer(x, mask=mask, freqs_cis=freqs_cis, context=context)
                
        return x
