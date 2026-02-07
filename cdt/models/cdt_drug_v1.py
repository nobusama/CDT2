"""
CDT-Drug v1 Model: Drug Response Prediction with Central Dogma Transformer

Based on CDT v3.3.1 architecture (v8 style with Flash Attention).

Input:
  - DNA: [batch, 896, 3072] - TSS-centered Enformer embeddings (static context)
  - Protein: [n_proteins, 768] - ProteomeLM embeddings (shared)
  - RNA: [batch, n_genes, 512] - scGPT cell-dependent embeddings
  - Drug: drug_idx [batch] + dose [batch] - Drug treatment info

Output:
  - gene_response: [batch, n_genes] - Per-gene response (log fold change)

Architecture (same as v8 + Drug):
  1. Projection (dimension unification)
  2. Drug Embedding (learned + dose encoding)
  3. Self-Attention (DNA, RNA, Protein + Drug)
  4. Cross-Attention (DNA → RNA → Protein)
  5. VCE Fusion Layer (Attention Pooling)
  6. Task Layer (gene response prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class CDTDrugConfig:
    """CDT-Drug v1 Model Configuration (v8 style)"""
    # Input dimensions
    dna_dim: int = 3072
    dna_seq_len: int = 896
    protein_dim: int = 768
    rna_dim: int = 512
    n_proteins: int = 2360
    n_genes: int = 2360

    # Drug parameters
    n_drugs: int = 188
    drug_embed_dim: int = 256

    # Model dimensions (same as v8)
    hidden_dim: int = 768
    nhead: int = 8
    dropout: float = 0.3

    # Self-Attention layers (same as v8)
    dna_self_attn_layers: int = 2
    rna_self_attn_layers: int = 1
    protein_self_attn_layers: int = 1


class SequenceProjector(nn.Module):
    """Same as v8"""
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


class DrugEmbedding(nn.Module):
    """Drug Embedding Module (CPA-style learned embeddings)"""

    def __init__(
        self,
        n_drugs: int,
        drug_embed_dim: int = 256,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.drug_embedding = nn.Embedding(n_drugs, drug_embed_dim)

        self.dose_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, drug_embed_dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(drug_embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, drug_idx: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        if dose.dim() == 1:
            dose = dose.unsqueeze(-1)
        drug_emb = self.drug_embedding(drug_idx)
        dose_emb = self.dose_encoder(dose)
        combined = torch.cat([drug_emb, dose_emb], dim=-1)
        return self.projector(combined)


class FlashSelfAttentionBlock(nn.Module):
    """Self-Attention with Flash Attention (same as v8)"""

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
    """Cross-Attention with Flash Attention (same as v8)"""

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
    """VCE with attention pooling (same as v8)"""

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


class CDTDrugModel(nn.Module):
    """
    CDT-Drug v1 Model (v8 style + Drug Embedding)

    Changes from v8:
    1. Added DrugEmbedding layer
    2. Drug modulation: Protein + Drug embedding
    3. Output: gene_response [batch, n_genes] instead of beta
    """

    def __init__(self, config: Optional[CDTDrugConfig] = None):
        super().__init__()
        if config is None:
            config = CDTDrugConfig()
        self.config = config

        # Projectors (same as v8)
        self.dna_projector = SequenceProjector(config.dna_dim, config.hidden_dim, config.dropout)
        self.rna_projector = SequenceProjector(config.rna_dim, config.hidden_dim, config.dropout)
        self.protein_projector = SequenceProjector(config.protein_dim, config.hidden_dim, config.dropout)

        # Drug Embedding (NEW)
        self.drug_embedding = DrugEmbedding(
            n_drugs=config.n_drugs,
            drug_embed_dim=config.drug_embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )

        # Self-Attention (same as v8)
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

        # Cross-Attention (same as v8)
        self.dna_to_rna = FlashCrossAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
        self.rna_to_protein = FlashCrossAttentionBlock(config.hidden_dim, config.nhead, config.dropout)

        # VCE (same as v8)
        self.vce = VirtualCellEmbedderWithAttention(config.hidden_dim, config.dropout)

        # Task Layer - predicts gene response [batch, n_genes]
        self.task_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_genes)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        rna_emb: torch.Tensor,
        drug_idx: torch.Tensor,
        dose: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dna_emb: [batch, 896, 3072] - TSS-centered Enformer
            protein_emb: [n_proteins, 768] - ProteomeLM (shared)
            rna_emb: [batch, n_genes, 512] - scGPT cell-dependent
            drug_idx: [batch] - Drug indices
            dose: [batch] - Dose values (log-scaled)

        Returns:
            gene_response: [batch, n_genes] - Predicted gene response
        """
        batch_size = dna_emb.size(0)

        # Projection
        dna = self.dna_projector(dna_emb)
        rna = self.rna_projector(rna_emb)
        protein = self.protein_projector(protein_emb)
        protein = protein.unsqueeze(0).expand(batch_size, -1, -1)

        # Drug Embedding
        drug_emb = self.drug_embedding(drug_idx, dose)  # [batch, hidden]

        # DNA Self-Attention
        for layer in self.dna_self_attn_layers:
            dna, _ = layer(dna)

        # RNA Self-Attention
        for layer in self.rna_self_attn_layers:
            rna, _ = layer(rna)

        # Drug modulation: Add drug to protein
        protein_with_drug = protein + drug_emb.unsqueeze(1)

        # Protein Self-Attention (with drug)
        for layer in self.protein_self_attn_layers:
            protein_with_drug, _ = layer(protein_with_drug)

        # Cross-Attention: DNA → RNA → Protein
        rna, _ = self.dna_to_rna(query=rna, key_value=dna)
        protein_with_drug, _ = self.rna_to_protein(query=protein_with_drug, key_value=rna)

        # VCE
        cell_embedding = self.vce(dna, rna, protein_with_drug)

        # Task Layer
        gene_response = self.task_layer(cell_embedding)

        return gene_response

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("CDT-Drug v1 Model Test (v8 style)")
    print("=" * 60)

    batch_size = 4
    n_proteins = 100
    n_genes = 100
    n_drugs = 20

    dna_emb = torch.randn(batch_size, 896, 3072)
    protein_emb = torch.randn(n_proteins, 768)
    rna_emb = torch.randn(batch_size, n_genes, 512)
    drug_idx = torch.randint(0, n_drugs, (batch_size,))
    dose = torch.rand(batch_size)

    config = CDTDrugConfig(n_proteins=n_proteins, n_genes=n_genes, n_drugs=n_drugs)
    model = CDTDrugModel(config)

    print(f"\nModel parameters: {model.get_num_params():,}")

    gene_response = model(dna_emb, protein_emb, rna_emb, drug_idx, dose)
    print(f"Output shape: {gene_response.shape}")

    print("\nTest passed!")
