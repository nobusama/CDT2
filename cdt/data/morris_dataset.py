"""
Morris STINGseq Dataset for CDT

Based on CDTRawExpressionDatasetV2 (v9) with modifications for Morris data:
- Input: Morris scRNA-seq expression (per-cell or pseudo-bulk)
- Output: 193 ADT protein levels (instead of beta values)

Key principles from v9 (must maintain):
1. Proper filtering of unmatched samples
2. CPM + log1p normalization for expression
3. Same gene embedding + expression projection architecture
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import gzip


class MorrisADTDataset(Dataset):
    """
    Morris STINGseq Dataset for CDT-Morris

    Predicts 193 ADT protein levels from:
    - DNA: TSS-centered Enformer embeddings [896, 3072]
    - RNA: Per-cell scRNA-seq expression [n_genes]

    Note: This is a cell-level dataset (unlike Gasperini which is enhancer-level)
    """

    def __init__(
        self,
        morris_h5_path: str,
        tss_dna_path: str,
        proteomelm_path: str,
        target_gene_list: List[str] = None,
        max_genes: int = 2360,
        batch_filter: List[str] = None,
        normalize_adt: bool = True,
    ):
        """
        Args:
            morris_h5_path: Path to Morris STINGseq H5 file with RNA + ADT data
            tss_dna_path: Path to TSS-centered Enformer embeddings
            proteomelm_path: Path to ProteomeLM embeddings
            target_gene_list: List of target gene names (will use intersection with Morris)
            max_genes: Maximum number of genes to use
            batch_filter: List of batch names to include (None = all)
            normalize_adt: Whether to normalize ADT values (CLR normalization)
        """
        self.morris_path = Path(morris_h5_path)
        self.tss_dna_path = Path(tss_dna_path)
        self.proteomelm_path = Path(proteomelm_path)
        self.normalize_adt = normalize_adt

        # Load Morris data
        self._load_morris_data(batch_filter)

        # Load ProteomeLM for gene-protein mapping
        self._load_proteomelm(target_gene_list, max_genes)

        # Load TSS DNA embeddings (if available)
        self._load_dna_embeddings()

        # Build gene index mapping (Morris genes -> target genes)
        self._build_gene_mapping()

        print(f"\nMorrisADTDataset initialized:")
        print(f"  Cells: {len(self)}")
        print(f"  RNA genes: {len(self.target_gene_names)}")
        print(f"  ADT proteins: {self.n_adt}")

    def _load_morris_data(self, batch_filter: Optional[List[str]]):
        """Load Morris STINGseq data"""
        print(f"Loading Morris data from {self.morris_path}...")

        with h5py.File(self.morris_path, 'r') as f:
            # Load gene information
            self.morris_gene_ids = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_ids'][:]
            ]
            self.morris_gene_names = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_names'][:]
            ]

            # Load batch information for filtering
            if 'batch' in f:
                all_batches = [
                    b.decode() if isinstance(b, bytes) else b
                    for b in f['batch'][:]
                ]
            else:
                all_batches = ['all'] * f.attrs.get('n_cells', len(f['cell_barcodes']))

            # Filter by batch if specified
            if batch_filter is not None:
                cell_mask = np.array([b in batch_filter for b in all_batches])
            else:
                cell_mask = np.ones(len(all_batches), dtype=bool)

            self.cell_indices = np.where(cell_mask)[0]
            self.batches = [all_batches[i] for i in self.cell_indices]

            # Load cell barcodes
            self.cell_barcodes = [
                f['cell_barcodes'][i].decode() if isinstance(f['cell_barcodes'][i], bytes)
                else f['cell_barcodes'][i]
                for i in self.cell_indices
            ]

            # Expression matrix info (will load on-the-fly)
            self.n_cells = len(self.cell_indices)
            self.n_morris_genes = len(self.morris_gene_names)

            # Check for sparse matrix format
            self.sparse_format = f.attrs.get('X_sparse', False)

            # Load ADT data if available
            if 'adt_counts' in f:
                self.adt_counts = f['adt_counts'][:][self.cell_indices]
                self.adt_names = [
                    a.decode() if isinstance(a, bytes) else a
                    for a in f['adt_names'][:]
                ]
                self.n_adt = len(self.adt_names)

                if self.normalize_adt:
                    self.adt_counts = self._clr_normalize(self.adt_counts)
            else:
                # ADT not yet integrated - set placeholder
                self.adt_counts = None
                self.adt_names = []
                self.n_adt = 0
                print("  Warning: ADT data not found in Morris H5 file")

        # Build gene name to index mapping for Morris data
        self.morris_gene_name_to_idx = {
            name: i for i, name in enumerate(self.morris_gene_names)
        }

        print(f"  Loaded: {self.n_cells} cells, {self.n_morris_genes} genes")
        if self.n_adt > 0:
            print(f"  ADT proteins: {self.n_adt}")

    def _clr_normalize(self, counts: np.ndarray) -> np.ndarray:
        """CLR (Centered Log Ratio) normalization for ADT counts"""
        # Add pseudocount
        counts = counts + 1

        # Log transform
        log_counts = np.log(counts)

        # Subtract geometric mean (per cell)
        geom_mean = log_counts.mean(axis=1, keepdims=True)
        clr = log_counts - geom_mean

        return clr.astype(np.float32)

    def _load_proteomelm(self, target_gene_list: Optional[List[str]], max_genes: int):
        """Load ProteomeLM embeddings to get target gene list"""
        print(f"Loading ProteomeLM from {self.proteomelm_path}...")

        with h5py.File(self.proteomelm_path, 'r') as f:
            self.protein_emb = f['embeddings'][:]
            self.protein_gene_names = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_names'][:]
            ]

        # Use provided list or ProteomeLM genes
        if target_gene_list is not None:
            self.target_gene_names = target_gene_list[:max_genes]
        else:
            self.target_gene_names = self.protein_gene_names[:max_genes]

        self.n_target_genes = len(self.target_gene_names)
        self.target_gene_to_idx = {
            name: i for i, name in enumerate(self.target_gene_names)
        }

        print(f"  Target genes: {self.n_target_genes}")

    def _load_dna_embeddings(self):
        """Load TSS-centered Enformer embeddings"""
        if not self.tss_dna_path.exists():
            print(f"  Warning: TSS DNA embeddings not found at {self.tss_dna_path}")
            self.dna_emb = None
            self.dna_gene_to_idx = {}
            return

        print(f"Loading TSS DNA embeddings from {self.tss_dna_path}...")

        with h5py.File(self.tss_dna_path, 'r') as f:
            self.dna_emb = f['embeddings'][:]
            dna_gene_names = [
                g.decode() if isinstance(g, bytes) else g
                for g in f['gene_names'][:]
            ]

        self.dna_gene_to_idx = {name: i for i, name in enumerate(dna_gene_names)}
        print(f"  Loaded: {len(dna_gene_names)} gene TSS embeddings")

    def _build_gene_mapping(self):
        """Build mapping from target gene indices to Morris gene indices"""
        self.target_to_morris_idx = {}

        for target_idx, gene_name in enumerate(self.target_gene_names):
            if gene_name in self.morris_gene_name_to_idx:
                morris_idx = self.morris_gene_name_to_idx[gene_name]
                self.target_to_morris_idx[target_idx] = morris_idx

        matched = len(self.target_to_morris_idx)
        total = len(self.target_gene_names)
        print(f"  Gene mapping: {matched}/{total} ({100*matched/total:.1f}%)")

    def _get_expression_vector(self, cell_idx: int) -> np.ndarray:
        """Get expression vector for a cell, mapped to target genes"""
        # Load expression for this cell (lazy loading for memory efficiency)
        with h5py.File(self.morris_path, 'r') as f:
            if self.sparse_format:
                # Reconstruct sparse row
                data = f['X_data'][:]
                indices = f['X_indices'][:]
                indptr = f['X_indptr'][:]
                shape = f.attrs['X_shape']

                actual_idx = self.cell_indices[cell_idx]
                start = indptr[actual_idx]
                end = indptr[actual_idx + 1]

                morris_expr = np.zeros(shape[1], dtype=np.float32)
                morris_expr[indices[start:end]] = data[start:end]
            else:
                actual_idx = self.cell_indices[cell_idx]
                morris_expr = f['X'][actual_idx, :].astype(np.float32)

        # Map to target genes
        target_expr = np.zeros(self.n_target_genes, dtype=np.float32)
        for target_idx, morris_idx in self.target_to_morris_idx.items():
            target_expr[target_idx] = morris_expr[morris_idx]

        # Normalize: CPM + log1p
        total = target_expr.sum()
        if total > 0:
            target_expr = target_expr * 10000 / total  # CPM
        target_expr = np.log1p(target_expr)

        return target_expr

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single cell sample

        Returns:
            dict: {
                'rna_expr': [n_target_genes] - RNA expression
                'adt_levels': [n_adt] - ADT protein levels (target)
                'cell_barcode': str - Cell identifier
            }
        """
        # RNA expression
        rna_expr = self._get_expression_vector(idx)

        # ADT levels (target)
        if self.adt_counts is not None:
            adt_levels = self.adt_counts[idx]
        else:
            adt_levels = np.zeros(193, dtype=np.float32)  # Placeholder

        return {
            'rna_expr': torch.from_numpy(rna_expr),
            'adt_levels': torch.from_numpy(adt_levels.astype(np.float32)),
            'batch': self.batches[idx],
        }

    def get_protein_embeddings(self) -> torch.Tensor:
        """Get ProteomeLM embeddings for target genes"""
        # Get embeddings only for target genes
        target_embs = []
        for gene_name in self.target_gene_names:
            if gene_name in self.protein_gene_names:
                idx = self.protein_gene_names.index(gene_name)
                target_embs.append(self.protein_emb[idx])
            else:
                # Zero embedding for unmatched genes
                target_embs.append(np.zeros(self.protein_emb.shape[1], dtype=np.float32))

        return torch.from_numpy(np.stack(target_embs).astype(np.float32))

    def get_shared_dna_emb(self) -> Optional[torch.Tensor]:
        """Get shared DNA embeddings (same for all cells)"""
        if self.dna_emb is None:
            return None

        # Get DNA embeddings for target genes
        dna_embs = []
        for gene_name in self.target_gene_names:
            if gene_name in self.dna_gene_to_idx:
                idx = self.dna_gene_to_idx[gene_name]
                dna_embs.append(self.dna_emb[idx])
            else:
                # Zero embedding for unmatched genes
                dna_embs.append(np.zeros((896, 3072), dtype=np.float32))

        return torch.from_numpy(np.stack(dna_embs).astype(np.float32))


def collate_morris(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for MorrisADTDataset"""
    rna_exprs = torch.stack([item['rna_expr'] for item in batch])
    adt_levels = torch.stack([item['adt_levels'] for item in batch])
    batches = [item['batch'] for item in batch]

    return {
        'rna_expr': rna_exprs,
        'adt_levels': adt_levels,
        'batch': batches,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Morris Dataset Test")
    print("=" * 60)

    # Test paths (update as needed)
    project_root = Path(__file__).parent.parent.parent

    morris_path = project_root / "data/processed/morris/stingseq_v2.h5"
    proteomelm_path = project_root / "data/processed/embeddings/human_proteomelm_embeddings_aligned.h5"
    tss_dna_path = project_root / "data/processed/embeddings/tss_enformer/morris_genes.h5"

    if morris_path.exists() and proteomelm_path.exists():
        dataset = MorrisADTDataset(
            morris_h5_path=str(morris_path),
            tss_dna_path=str(tss_dna_path),
            proteomelm_path=str(proteomelm_path),
            max_genes=100,  # Small test
        )

        print(f"\nDataset length: {len(dataset)}")

        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  RNA expr shape: {sample['rna_expr'].shape}")
        print(f"  ADT levels shape: {sample['adt_levels'].shape}")
        print(f"  Batch: {sample['batch']}")
    else:
        print("Test files not found, skipping test")
