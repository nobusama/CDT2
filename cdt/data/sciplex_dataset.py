"""
sci-Plex Drug Response Dataset for CDT-Drug

Handles loading and batching of sci-Plex K562 drug perturbation data
for training the CDT-Drug model.

Data structure:
  - rna_embeddings: [n_cells, n_genes, 512] - scGPT cell-dependent embeddings
  - dna_context: [n_genes, 896, 3072] - TSS-centered Enformer embeddings (shared)
  - drug_idx: [n_cells] - Drug indices
  - dose: [n_cells] - Dose values (log-scaled)
  - gene_response: [n_cells, n_genes] - Target gene response (log fold change)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import scipy.sparse as sp


class SciPlexDrugDataset(Dataset):
    """
    sci-Plex Drug Response Dataset

    Loads pre-processed sci-Plex K562 data for CDT-Drug training.
    """

    def __init__(
        self,
        data_path: Path,
        embeddings_path: Path,
        dna_embeddings_path: Optional[Path] = None,
        protein_embeddings_path: Optional[Path] = None,
        mode: str = 'train',
        max_cells: Optional[int] = None
    ):
        """
        Args:
            data_path: Path to sci-Plex H5 file (k562_train.h5, k562_val.h5, etc.)
            embeddings_path: Path to RNA embeddings H5 file
            dna_embeddings_path: Path to TSS Enformer embeddings (optional, can be shared)
            protein_embeddings_path: Path to protein embeddings (optional, can be shared)
            mode: 'train', 'val', or 'test'
            max_cells: Maximum number of cells to use (for debugging)
        """
        self.mode = mode
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)

        # Load data
        self._load_data(max_cells)

        # Load embeddings
        self._load_embeddings(dna_embeddings_path, protein_embeddings_path)

    def _load_data(self, max_cells: Optional[int] = None):
        """Load cell data and metadata"""
        print(f"Loading sci-Plex {self.mode} data from {self.data_path}")

        with h5py.File(self.data_path, 'r') as f:
            # Drug and dose info
            self.drug_idx = f['drug_idx'][:]
            self.dose = f['dose_log'][:]  # Log-scaled dose
            self.is_control = f['is_control'][:]

            # Expression matrix (for gene response computation)
            if 'X_data' in f:
                # Sparse matrix
                data = f['X_data'][:]
                indices = f['X_indices'][:]
                indptr = f['X_indptr'][:]
                shape = f.attrs['X_shape']
                self.expression = sp.csr_matrix((data, indices, indptr), shape=shape)
            else:
                self.expression = f['X'][:]

            # Gene names
            self.gene_names = [g.decode() if isinstance(g, bytes) else g
                             for g in f['gene_names'][:]]

            # Metadata
            self.n_drugs = f.attrs.get('n_drugs', self.drug_idx.max() + 1)

        self.n_cells = len(self.drug_idx)
        self.n_genes = len(self.gene_names)

        # Limit cells if specified
        if max_cells is not None and max_cells < self.n_cells:
            self.n_cells = max_cells
            self.drug_idx = self.drug_idx[:max_cells]
            self.dose = self.dose[:max_cells]
            self.is_control = self.is_control[:max_cells]
            if sp.issparse(self.expression):
                self.expression = self.expression[:max_cells]
            else:
                self.expression = self.expression[:max_cells]

        print(f"  Cells: {self.n_cells}")
        print(f"  Genes: {self.n_genes}")
        print(f"  Drugs: {self.n_drugs}")

    def _load_embeddings(
        self,
        dna_path: Optional[Path] = None,
        protein_path: Optional[Path] = None
    ):
        """Load pre-computed embeddings"""
        # RNA embeddings (cell-dependent)
        print(f"Loading RNA embeddings from {self.embeddings_path}")

        self.rna_file = h5py.File(self.embeddings_path, 'r')
        self.rna_embeddings = self.rna_file['embeddings']  # Lazy load: [n_cells, n_genes, 512]

        print(f"  RNA embeddings shape: {self.rna_embeddings.shape}")

        # DNA embeddings (optional, shared across all cells)
        if dna_path is not None:
            print(f"Loading DNA embeddings from {dna_path}")
            with h5py.File(dna_path, 'r') as f:
                self.dna_embeddings = f['embeddings'][:]  # [n_genes, 896, 3072]
            print(f"  DNA embeddings shape: {self.dna_embeddings.shape}")
        else:
            self.dna_embeddings = None

        # Protein embeddings (optional, shared)
        if protein_path is not None:
            print(f"Loading protein embeddings from {protein_path}")
            with h5py.File(protein_path, 'r') as f:
                self.protein_embeddings = f['embeddings'][:]  # [n_proteins, 768]
            print(f"  Protein embeddings shape: {self.protein_embeddings.shape}")
        else:
            self.protein_embeddings = None

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single cell sample.

        Returns:
            Dict with:
                - rna_emb: [n_genes, 512] - Cell-dependent RNA embeddings
                - drug_idx: scalar - Drug index
                - dose: scalar - Log-scaled dose
                - gene_response: [n_genes] - Target gene response (placeholder)
        """
        # RNA embeddings for this cell
        rna_emb = self.rna_embeddings[idx]  # [n_genes, 512]
        if isinstance(rna_emb, np.ndarray):
            rna_emb = torch.from_numpy(rna_emb).float()
        else:
            rna_emb = torch.tensor(rna_emb, dtype=torch.float32)

        # Drug info
        drug_idx = torch.tensor(self.drug_idx[idx], dtype=torch.long)
        dose = torch.tensor(self.dose[idx], dtype=torch.float32)

        # Gene response (placeholder - needs to be computed from DEG analysis)
        # For now, use normalized expression as proxy
        if sp.issparse(self.expression):
            expr = self.expression[idx].toarray().flatten()
        else:
            expr = self.expression[idx]
        gene_response = torch.tensor(expr, dtype=torch.float32)
        gene_response = torch.log1p(gene_response)  # Log transform

        return {
            'rna_emb': rna_emb,
            'drug_idx': drug_idx,
            'dose': dose,
            'gene_response': gene_response,
            'is_control': torch.tensor(self.is_control[idx], dtype=torch.bool)
        }

    def get_shared_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get shared embeddings (DNA, Protein) that are constant across batch.

        Returns:
            Dict with:
                - dna_emb: [n_genes, 896, 3072] if available
                - protein_emb: [n_proteins, 768] if available
        """
        result = {}

        if self.dna_embeddings is not None:
            result['dna_emb'] = torch.from_numpy(self.dna_embeddings).float()

        if self.protein_embeddings is not None:
            result['protein_emb'] = torch.from_numpy(self.protein_embeddings).float()

        return result

    def close(self):
        """Close H5 files"""
        if hasattr(self, 'rna_file') and self.rna_file:
            self.rna_file.close()


class SciPlexDEGDataset(Dataset):
    """
    sci-Plex Dataset with pre-computed DEG (Differentially Expressed Genes).

    Uses pre-computed log fold changes relative to DMSO controls.
    """

    def __init__(
        self,
        deg_path: Path,
        embeddings_path: Path,
        mode: str = 'train'
    ):
        """
        Args:
            deg_path: Path to H5 file with DEG data
            embeddings_path: Path to RNA embeddings
            mode: 'train', 'val', or 'test'
        """
        self.mode = mode

        print(f"Loading DEG data from {deg_path}")
        with h5py.File(deg_path, 'r') as f:
            self.drug_idx = f['drug_idx'][:]
            self.dose = f['dose_log'][:]
            self.log_fc = f['log_fc'][:]  # [n_samples, n_genes] - Log fold change
            self.gene_names = [g.decode() if isinstance(g, bytes) else g
                              for g in f['gene_names'][:]]

        self.n_samples = len(self.drug_idx)
        self.n_genes = len(self.gene_names)

        # Load RNA embeddings
        print(f"Loading RNA embeddings from {embeddings_path}")
        self.rna_file = h5py.File(embeddings_path, 'r')
        self.rna_embeddings = self.rna_file['embeddings']

        print(f"  Samples: {self.n_samples}")
        print(f"  Genes: {self.n_genes}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rna_emb = torch.from_numpy(self.rna_embeddings[idx]).float()
        drug_idx = torch.tensor(self.drug_idx[idx], dtype=torch.long)
        dose = torch.tensor(self.dose[idx], dtype=torch.float32)
        log_fc = torch.from_numpy(self.log_fc[idx]).float()

        return {
            'rna_emb': rna_emb,
            'drug_idx': drug_idx,
            'dose': dose,
            'gene_response': log_fc
        }

    def close(self):
        if hasattr(self, 'rna_file') and self.rna_file:
            self.rna_file.close()


def create_dataloaders(
    data_dir: Path,
    embeddings_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.

    Args:
        data_dir: Directory with k562_train.h5, k562_val.h5, k562_test.h5
        embeddings_dir: Directory with RNA/DNA/Protein embeddings
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    embeddings_dir = Path(embeddings_dir)

    # Create datasets
    train_dataset = SciPlexDrugDataset(
        data_path=data_dir / "k562_train.h5",
        embeddings_path=embeddings_dir / "k562_sciplex_rna_embeddings.h5",
        dna_embeddings_path=embeddings_dir / "tss_enformer_embeddings.h5",
        protein_embeddings_path=embeddings_dir / "human_proteomelm_embeddings_aligned.h5",
        mode='train'
    )

    val_dataset = SciPlexDrugDataset(
        data_path=data_dir / "k562_val.h5",
        embeddings_path=embeddings_dir / "k562_sciplex_rna_embeddings.h5",
        dna_embeddings_path=embeddings_dir / "tss_enformer_embeddings.h5",
        protein_embeddings_path=embeddings_dir / "human_proteomelm_embeddings_aligned.h5",
        mode='val'
    )

    test_dataset = SciPlexDrugDataset(
        data_path=data_dir / "k562_test.h5",
        embeddings_path=embeddings_dir / "k562_sciplex_rna_embeddings.h5",
        dna_embeddings_path=embeddings_dir / "tss_enformer_embeddings.h5",
        protein_embeddings_path=embeddings_dir / "human_proteomelm_embeddings_aligned.h5",
        mode='test'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("SciPlex Dataset Test")
    print("=" * 40)

    # Test with dummy data
    import tempfile

    # Create dummy H5 files for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dummy data file
        n_cells = 100
        n_genes = 50

        with h5py.File(tmpdir / "test_data.h5", 'w') as f:
            f.create_dataset('drug_idx', data=np.random.randint(0, 10, n_cells))
            f.create_dataset('dose_log', data=np.random.rand(n_cells))
            f.create_dataset('is_control', data=np.random.rand(n_cells) < 0.1)
            f.create_dataset('X', data=np.random.rand(n_cells, n_genes))
            f.create_dataset('gene_names', data=np.array([f"Gene{i}" for i in range(n_genes)], dtype='S'))
            f.attrs['n_drugs'] = 10

        # Create dummy embeddings file
        with h5py.File(tmpdir / "test_emb.h5", 'w') as f:
            f.create_dataset('embeddings', data=np.random.rand(n_cells, n_genes, 512).astype(np.float32))

        # Test dataset
        dataset = SciPlexDrugDataset(
            data_path=tmpdir / "test_data.h5",
            embeddings_path=tmpdir / "test_emb.h5",
            mode='test',
            max_cells=50
        )

        print(f"\nDataset size: {len(dataset)}")

        # Get sample
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

        dataset.close()
        print("\nTest complete!")
