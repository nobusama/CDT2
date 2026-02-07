"""
Morris CRISPRi Dataset for CDT v2

Predicts CRISPRi perturbation effects on gene expression.

Input:
  - DNA: Enformer embedding for target gene TSS [896, 3072]
  - RNA: Baseline expression profile [2360 genes]

Output:
  - CRISPRi effect: log2 fold change on 2360 genes (or all 36601 genes)

This dataset is PERTURBATION-level (not cell-level like ADT):
  - Each sample = one perturbation (targeting one gene)
  - Input = shared DNA embedding + shared baseline expression
  - Output = effect on all genes
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MorrisCRISPRiDataset(Dataset):
    """
    Morris CRISPRi Effect Prediction Dataset for CDT v2

    Predicts the effect of CRISPRi knockdown on gene expression.
    Each sample is one perturbation targeting a specific gene.

    Architecture:
        DNA (target gene TSS) + RNA (baseline) → Model → CRISPRi effects
    """

    def __init__(
        self,
        effects_h5_path: str,
        enformer_h5_path: str,
        baseline_expr_h5_path: Optional[str] = None,
        output_genes: int = 2360,
        gene_list_path: Optional[str] = None,
    ):
        """
        Args:
            effects_h5_path: Path to CRISPRi effects H5 file (from effect calculation)
            enformer_h5_path: Path to Enformer embeddings for target genes
            baseline_expr_h5_path: Path to baseline expression (optional, use NTC mean)
            output_genes: Number of output genes (2360 for CDT v2, or all for full)
            gene_list_path: Path to gene list file (for 2360 gene mapping)
        """
        self.effects_path = Path(effects_h5_path)
        self.enformer_path = Path(enformer_h5_path)
        self.output_genes = output_genes

        # Load effect sizes
        self._load_effects()

        # Load Enformer embeddings
        self._load_enformer()

        # Load or create baseline expression
        if baseline_expr_h5_path:
            self._load_baseline(baseline_expr_h5_path)
        else:
            self._create_baseline_from_effects()

        # Build gene mapping (Morris 36601 → CDT 2360)
        if gene_list_path:
            self._load_gene_list(gene_list_path)
        else:
            self.gene_mapping = None  # Use all genes

        logger.info(f"MorrisCRISPRiDataset initialized:")
        logger.info(f"  Perturbations: {len(self)}")
        logger.info(f"  Input genes (RNA): {self.n_input_genes}")
        logger.info(f"  Output genes (effect): {self.n_output_genes}")

    def _load_effects(self):
        """Load CRISPRi effect sizes"""
        logger.info(f"Loading effects from {self.effects_path}")

        with h5py.File(self.effects_path, 'r') as f:
            # log2FC matrix: [n_perturbations, n_genes]
            self.log2fc = f['log2fc'][:]

            # Target gene names (perturbation targets)
            self.target_genes = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['target_genes'][:]
            ]

            # Output gene names
            self.output_gene_names = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['output_genes'][:]
            ]

            # Metadata
            self.self_effects = f['self_effects'][:]
            self.n_cells = f['n_cells'][:]
            self.baselines = f['baselines'][:]

        self.n_perturbations = len(self.target_genes)
        self.n_output_genes = self.log2fc.shape[1]

        logger.info(f"  {self.n_perturbations} perturbations, {self.n_output_genes} output genes")

    def _load_enformer(self):
        """Load Enformer embeddings for target genes"""
        logger.info(f"Loading Enformer from {self.enformer_path}")

        with h5py.File(self.enformer_path, 'r') as f:
            self.enformer_emb = f['embeddings'][:]  # [n_genes, 896, 3072]
            enformer_genes = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_names'][:]
            ]

        # Build mapping: target gene → Enformer index
        self.target_to_enformer = {}
        for i, gene in enumerate(self.target_genes):
            if gene in enformer_genes:
                self.target_to_enformer[i] = enformer_genes.index(gene)
            else:
                logger.warning(f"  Target gene {gene} not found in Enformer embeddings")

        logger.info(f"  {len(self.target_to_enformer)}/{len(self.target_genes)} targets have Enformer")

    def _create_baseline_from_effects(self):
        """Create baseline expression from NTC mean (stored in effects file)"""
        # For now, use zeros as baseline (will be loaded from Morris data in training)
        self.baseline_expr = np.zeros(self.n_output_genes, dtype=np.float32)
        self.n_input_genes = self.n_output_genes
        logger.info(f"  Using zero baseline (will load from Morris data)")

    def _load_baseline(self, path: str):
        """Load baseline expression from file"""
        logger.info(f"Loading baseline from {path}")
        with h5py.File(path, 'r') as f:
            self.baseline_expr = f['expression'][:]
        self.n_input_genes = len(self.baseline_expr)

    def _load_gene_list(self, path: str):
        """Load CDT gene list for output mapping"""
        logger.info(f"Loading gene list from {path}")
        with h5py.File(path, 'r') as f:
            cdt_genes = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_names'][:]
            ]

        # Build mapping: CDT gene index → Morris gene index
        self.gene_mapping = {}
        morris_gene_to_idx = {g: i for i, g in enumerate(self.output_gene_names)}

        for i, gene in enumerate(cdt_genes[:self.output_genes]):
            if gene in morris_gene_to_idx:
                self.gene_mapping[i] = morris_gene_to_idx[gene]

        logger.info(f"  {len(self.gene_mapping)}/{self.output_genes} genes mapped")

    def __len__(self) -> int:
        return self.n_perturbations

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single perturbation sample

        Returns:
            dict: {
                'dna_emb': [896, 3072] - Enformer embedding for target gene TSS
                'rna_expr': [n_input_genes] - Baseline RNA expression
                'effect': [n_output_genes] - CRISPRi effect (log2FC)
                'target_gene': str - Name of targeted gene
                'self_effect': float - Effect on the target gene itself
            }
        """
        target_gene = self.target_genes[idx]

        # DNA embedding
        if idx in self.target_to_enformer:
            enf_idx = self.target_to_enformer[idx]
            dna_emb = self.enformer_emb[enf_idx]  # [896, 3072]
        else:
            # Zero embedding if not available
            dna_emb = np.zeros((896, 3072), dtype=np.float32)

        # RNA baseline expression
        rna_expr = self.baseline_expr.copy()

        # CRISPRi effect (log2FC)
        if self.gene_mapping is not None:
            # Map to CDT gene subset
            effect = np.zeros(self.output_genes, dtype=np.float32)
            for cdt_idx, morris_idx in self.gene_mapping.items():
                effect[cdt_idx] = self.log2fc[idx, morris_idx]
        else:
            effect = self.log2fc[idx]

        return {
            'dna_emb': torch.from_numpy(dna_emb.astype(np.float32)),
            'rna_expr': torch.from_numpy(rna_expr.astype(np.float32)),
            'effect': torch.from_numpy(effect.astype(np.float32)),
            'target_gene': target_gene,
            'self_effect': torch.tensor(self.self_effects[idx], dtype=torch.float32),
            'baseline': torch.tensor(self.baselines[idx], dtype=torch.float32),
        }

    def get_target_gene_idx(self, idx: int) -> int:
        """Get the index of the target gene in the output gene list"""
        target_gene = self.target_genes[idx]
        if target_gene in self.output_gene_names:
            return self.output_gene_names.index(target_gene)
        return -1


def collate_crispri(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for MorrisCRISPRiDataset"""
    return {
        'dna_emb': torch.stack([item['dna_emb'] for item in batch]),
        'rna_expr': torch.stack([item['rna_expr'] for item in batch]),
        'effect': torch.stack([item['effect'] for item in batch]),
        'target_genes': [item['target_gene'] for item in batch],
        'self_effects': torch.stack([item['self_effect'] for item in batch]),
        'baselines': torch.stack([item['baseline'] for item in batch]),
    }


class MorrisCRISPRiDatasetV2(Dataset):
    """
    Simplified CRISPRi Dataset for CDT v2 Training

    Designed for Colab/GPU training with pre-loaded data.
    Uses shared DNA embedding and baseline expression.
    """

    def __init__(
        self,
        log2fc: np.ndarray,           # [n_perturbations, n_output_genes]
        target_genes: List[str],       # [n_perturbations]
        enformer_emb: np.ndarray,      # [n_perturbations, 896, 3072]
        baseline_expr: np.ndarray,     # [n_output_genes]
        self_effects: np.ndarray = None,
    ):
        """
        Args:
            log2fc: CRISPRi effects matrix
            target_genes: Names of targeted genes
            enformer_emb: Enformer embeddings for each target
            baseline_expr: Baseline (NTC) expression profile
            self_effects: Self-knockdown effects (optional)
        """
        self.log2fc = log2fc
        self.target_genes = target_genes
        self.enformer_emb = enformer_emb
        self.baseline_expr = baseline_expr
        self.self_effects = self_effects if self_effects is not None else np.zeros(len(target_genes))

        assert len(log2fc) == len(target_genes) == len(enformer_emb)

    def __len__(self) -> int:
        return len(self.target_genes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'dna_emb': torch.from_numpy(self.enformer_emb[idx].astype(np.float32)),
            'rna_expr': torch.from_numpy(self.baseline_expr.astype(np.float32)),
            'effect': torch.from_numpy(self.log2fc[idx].astype(np.float32)),
            'target_gene': self.target_genes[idx],
            'self_effect': torch.tensor(self.self_effects[idx], dtype=torch.float32),
        }


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Morris CRISPRi Dataset Test")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.parent

    effects_path = project_root / "data/processed/morris/crispri_effects.h5"
    enformer_path = project_root / "data/processed/morris/morris_28genes_enformer.h5"

    if effects_path.exists():
        # Note: Enformer file may not exist yet (needs Colab generation)
        print(f"\nEffects file found: {effects_path}")

        with h5py.File(effects_path, 'r') as f:
            print(f"  Target genes: {f.attrs['n_target_genes']}")
            print(f"  Output genes: {f.attrs['n_output_genes']}")
            print(f"  NTC cells: {f.attrs['n_ntc_cells']}")
            print(f"  Method: {f.attrs['method']}")

            target_genes = [g.decode() for g in f['target_genes'][:]]
            print(f"\n  Targets: {target_genes}")
    else:
        print(f"Effects file not found: {effects_path}")
