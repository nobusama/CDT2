"""
CDT-Morris Model: ADT Protein Prediction from scRNA-seq

Based on CDTRawModel (v3.5 Raw Expression v2) with modifications:
- Output: 193 ADT protein levels (instead of 2360 gene beta values)
- Input: Morris scRNA-seq (per-cell or pseudo-bulk)

Architecture (same as CDTRawModel):
1. RawExpressionEncoder: RNA expression -> [n_genes, hidden_dim]
2. Self-Attention: DNA(2 layers), RNA(1 layer), Protein(1 layer)
3. Cross-Attention: DNA -> RNA -> Protein
4. VCE: Attention pooling to fuse modalities
5. Task Layer: Predict ADT levels [batch, n_adt]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class CDTMorrisConfig:
    """CDT-Morris Model Configuration"""
    # Input dimensions
    dna_dim: int = 3072
    dna_seq_len: int = 896
    protein_dim: int = 768
    n_genes: int = 2360  # Number of RNA genes

    # Output dimension
    n_adt: int = 193  # Number of ADT proteins to predict

    # Model dimensions
    hidden_dim: int = 768
    nhead: int = 8
    dropout: float = 0.3

    # Self-Attention layers
    dna_self_attn_layers: int = 2
    rna_self_attn_layers: int = 1
    protein_self_attn_layers: int = 1


class RawExpressionEncoder(nn.Module):
    """
    Raw expression -> hidden_dim embeddings

    Each gene gets:
    - A learned gene identity embedding
    - A projection of its expression value
    """
    def __init__(self, n_genes: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Gene identity embedding (learned)
        self.gene_embedding = nn.Embedding(n_genes, hidden_dim)

        # Expression value projection
        self.expr_projector = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Combine gene identity and expression
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expression: [batch, n_genes] raw expression values

        Returns:
            embeddings: [batch, n_genes, hidden_dim]
        """
        batch_size = expression.size(0)
        device = expression.device

        # Gene identity embeddings (same for all samples)
        gene_ids = torch.arange(self.n_genes, device=device)
        gene_emb = self.gene_embedding(gene_ids)  # [n_genes, hidden_dim]
        gene_emb = gene_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Expression value embeddings
        expr_emb = self.expr_projector(expression.unsqueeze(-1))

        # Combine
        combined = torch.cat([gene_emb, expr_emb], dim=-1)
        output = self.combine(combined)

        return output


class SequenceProjector(nn.Module):
    """Projector for DNA and Protein embeddings"""
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class FlashSelfAttentionBlock(nn.Module):
    """Self-Attention with Flash Attention"""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, None


class FlashCrossAttentionBlock(nn.Module):
    """Cross-Attention with Flash Attention"""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        batch_size, query_len, _ = query.shape
        key_len = key_value.shape[1]

        Q = self.q_proj(query).view(batch_size, query_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(batch_size, key_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, None


class VirtualCellEmbedderWithAttention(nn.Module):
    """VCE with attention pooling (same as v3.3.1)"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = 4
        self.head_dim = d_model // self.nhead

        self.dna_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.rna_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.protein_query = nn.Parameter(torch.randn(1, 1, d_model))

        # DNA attention
        self.dna_q_proj = nn.Linear(d_model, d_model)
        self.dna_k_proj = nn.Linear(d_model, d_model)
        self.dna_v_proj = nn.Linear(d_model, d_model)
        self.dna_out_proj = nn.Linear(d_model, d_model)

        # RNA attention
        self.rna_q_proj = nn.Linear(d_model, d_model)
        self.rna_k_proj = nn.Linear(d_model, d_model)
        self.rna_v_proj = nn.Linear(d_model, d_model)
        self.rna_out_proj = nn.Linear(d_model, d_model)

        # Protein attention
        self.protein_q_proj = nn.Linear(d_model, d_model)
        self.protein_k_proj = nn.Linear(d_model, d_model)
        self.protein_v_proj = nn.Linear(d_model, d_model)
        self.protein_out_proj = nn.Linear(d_model, d_model)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def _attention_pool(self, query, key_value, q_proj, k_proj, v_proj, out_proj):
        batch_size = key_value.size(0)
        seq_len = key_value.size(1)
        query = query.expand(batch_size, -1, -1)

        Q = q_proj(query).view(batch_size, 1, self.nhead, self.head_dim).transpose(1, 2)
        K = k_proj(key_value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = v_proj(key_value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = out_proj(attn_out)
        return attn_out.squeeze(1)

    def forward(self, dna_encoded, rna_encoded, protein_encoded):
        dna_pooled = self._attention_pool(
            self.dna_query, dna_encoded,
            self.dna_q_proj, self.dna_k_proj, self.dna_v_proj, self.dna_out_proj
        )
        rna_pooled = self._attention_pool(
            self.rna_query, rna_encoded,
            self.rna_q_proj, self.rna_k_proj, self.rna_v_proj, self.rna_out_proj
        )
        protein_pooled = self._attention_pool(
            self.protein_query, protein_encoded,
            self.protein_q_proj, self.protein_k_proj, self.protein_v_proj, self.protein_out_proj
        )

        concat = torch.cat([dna_pooled, rna_pooled, protein_pooled], dim=-1)
        cell_embedding = self.fusion(concat)
        return cell_embedding


class CDTMorrisModel(nn.Module):
    """
    CDT-Morris Model for ADT Protein Prediction

    Input:
        - dna_emb: [batch, 896, 3072] or [n_genes, 896, 3072] (shared)
        - protein_emb: [n_proteins, 768] (shared ProteomeLM)
        - rna_expr: [batch, n_genes] raw expression values

    Output:
        - adt_pred: [batch, n_adt] predicted ADT protein levels
    """

    def __init__(self, config: Optional[CDTMorrisConfig] = None):
        super().__init__()
        if config is None:
            config = CDTMorrisConfig()
        self.config = config

        # Projectors
        self.dna_projector = SequenceProjector(config.dna_dim, config.hidden_dim, config.dropout)
        self.protein_projector = SequenceProjector(config.protein_dim, config.hidden_dim, config.dropout)
        self.rna_encoder = RawExpressionEncoder(config.n_genes, config.hidden_dim, config.dropout)

        # Self-Attention layers
        self.dna_self_attn_layers = nn.ModuleList([
            FlashSelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.dna_self_attn_layers)
        ])
        self.rna_self_attn_layers = nn.ModuleList([
            FlashSelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.rna_self_attn_layers)
        ])
        self.protein_self_attn_layers = nn.ModuleList([
            FlashSelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.protein_self_attn_layers)
        ])

        # Cross-Attention (Central Dogma flow)
        self.dna_to_rna = FlashCrossAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
        self.rna_to_protein = FlashCrossAttentionBlock(config.hidden_dim, config.nhead, config.dropout)

        # VCE
        self.vce = VirtualCellEmbedderWithAttention(config.hidden_dim, config.dropout)

        # Task Layer - predicts ADT levels [batch, n_adt]
        self.task_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_adt)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        rna_expr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dna_emb: [batch, 896, 3072] or [n_genes, 896, 3072] TSS-centered Enformer
            protein_emb: [n_proteins, 768] ProteomeLM (shared)
            rna_expr: [batch, n_genes] raw expression values

        Returns:
            adt_pred: [batch, n_adt] predicted ADT protein levels
        """
        batch_size = rna_expr.size(0)

        # Project DNA
        if dna_emb.dim() == 3 and dna_emb.size(0) == batch_size:
            # Per-sample DNA: [batch, 896, 3072]
            dna = self.dna_projector(dna_emb)
        else:
            # Shared DNA: [n_genes, 896, 3072] -> use mean or pooled
            dna = self.dna_projector(dna_emb.mean(dim=0, keepdim=True))
            dna = dna.expand(batch_size, -1, -1)

        # Encode RNA expression
        rna = self.rna_encoder(rna_expr)  # [batch, n_genes, hidden_dim]

        # Project Protein
        protein = self.protein_projector(protein_emb)  # [n_proteins, hidden_dim]
        protein = protein.unsqueeze(0).expand(batch_size, -1, -1)

        # DNA Self-Attention
        for layer in self.dna_self_attn_layers:
            dna, _ = layer(dna)

        # RNA Self-Attention
        for layer in self.rna_self_attn_layers:
            rna, _ = layer(rna)

        # Protein Self-Attention
        for layer in self.protein_self_attn_layers:
            protein, _ = layer(protein)

        # Cross-Attention: DNA -> RNA -> Protein (Central Dogma)
        rna, _ = self.dna_to_rna(query=rna, key_value=dna)
        protein, _ = self.rna_to_protein(query=protein, key_value=rna)

        # VCE fusion
        cell_embedding = self.vce(dna, rna, protein)

        # Task layer: predict ADT levels
        adt_pred = self.task_layer(cell_embedding)

        return adt_pred

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("CDT-Morris Model Test")
    print("=" * 60)

    # Test configuration
    batch_size = 4
    n_proteins = 100  # Smaller for testing
    n_genes = 100
    n_adt = 193

    # Create dummy inputs
    dna_emb = torch.randn(batch_size, 896, 3072)
    protein_emb = torch.randn(n_proteins, 768)
    rna_expr = torch.randn(batch_size, n_genes)

    # Create model
    config = CDTMorrisConfig(
        n_genes=n_genes,
        n_adt=n_adt
    )
    model = CDTMorrisModel(config)

    print(f"\nModel configuration:")
    print(f"  n_genes: {config.n_genes}")
    print(f"  n_adt: {config.n_adt}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  Parameters: {model.get_num_params():,}")

    # Forward pass
    adt_pred = model(dna_emb, protein_emb, rna_expr)

    print(f"\nInput shapes:")
    print(f"  DNA: {dna_emb.shape}")
    print(f"  Protein: {protein_emb.shape}")
    print(f"  RNA expr: {rna_expr.shape}")
    print(f"\nOutput shape: {adt_pred.shape}")

    print("\nTest passed!")
