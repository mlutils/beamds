from torch import nn
import torch
import math
import torch.nn.functional as F



class CausalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super(CausalConv, self).__init__()

        padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        b, f, l = x.shape
        x = self.conv(x)[:, :, :l]

        return x


class ResCausalConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, dilation=None):
        super(ResCausalConvBlock, self).__init__()

        self.conv1 = CausalConv(input_channels, output_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.gelu = nn.GELU()
        self.conv2 = CausalConv(output_channels, output_channels, kernel_size=kernel_size, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(output_channels)

        if input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(output_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Main transformation path
        r = self.conv1(x)
        r = self.bn1(r)
        r = self.gelu(r)
        r = self.conv2(r)
        r = self.bn2(r)

        # Shortcut path
        x = self.shortcut(x)

        # Combine transformation and shortcut, then apply GELU (Post-Addition Non-Linearity)
        out = r + x
        out = self.gelu(out)
        return out

class ResCausalConv(nn.Module):

    def __init__(self, input_channels, channels, kernel_sizes, dilations=None, bias=True):
        super(ResCausalConv, self).__init__()

        layers = [nn.BatchNorm1d(input_channels), nn.GELU()]

        prev_channels = input_channels

        for i in range(len(channels)):
            kernel_size = kernel_sizes[i]
            dilation = dilations[i] if dilations else 1
            layers.append(ResCausalConvBlock(prev_channels, channels[i], kernel_size, dilation))
            prev_channels = channels[i]

        self.resnet = nn.Sequential(*layers)

    def forward(self, x):
        x = self.resnet(x.transpose(1, 2))
        return x.transpose(1, 2)


def apply_rotary_positional_embedding(x, sin, cos):

    # Split x into even and odd indices
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices

    # Apply RoPE
    rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated_x


def get_rotary_emb(head_dim, seq_len, device):
    # Generate rotary positional embeddings
    position = torch.arange(0, seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, head_dim // 2, device=device, dtype=torch.float32)
    div_term = 1.0 / (10000 ** (2 * index / head_dim))
    angle = position * div_term
    sin, cos = torch.sin(angle), torch.cos(angle)
    return sin.unsqueeze(0).unsqueeze(2), cos.unsqueeze(0).unsqueeze(2)


class DecoderLayerWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin, cos):
        # Apply RoPE to query and key vectors
        bsz, seq_len, embed_dim = x.size()
        x = self.norm1(x)

        # Generate query, key, and value with RoPE
        qkv = x.view(bsz, seq_len, -1, embed_dim // self.self_attn.num_heads)
        q = apply_rotary_positional_embedding(qkv, sin, cos)
        k = apply_rotary_positional_embedding(qkv, sin, cos)

        # Reshape back for attention computation
        q = q.view(bsz, seq_len, embed_dim)
        k = k.view(bsz, seq_len, embed_dim)

        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        # Compute self-attention with causal mask
        attn_output, _ = self.self_attn(q, k, x, attn_mask=causal_mask)
        x = x + self.dropout(attn_output)

        # Feedforward
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)
        return x


class DecoderWithRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayerWithRoPE(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        device = x.device
        seq_len = x.size(1)
        head_dim = self.embed_dim // self.num_heads  # Calculate head_dim

        # Input Embedding
        x = x * math.sqrt(self.embed_dim)

        # Generate RoPE
        sin, cos = get_rotary_emb(head_dim, seq_len, device)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, sin, cos)

        # Output projection
        x = self.norm(x)
        return x