# Central Dogma Transformer II (CDT-II)

**An AI Microscope for Understanding Cellular Regulatory Mechanisms**

## Overview

CDT-II is an "AI microscope" whose attention maps are directly interpretable as regulatory structure. By mirroring the central dogma in its architecture, each attention mechanism corresponds to a specific biological relationship:

- **DNA self-attention**: Genomic relationships
- **RNA self-attention**: Gene co-regulation
- **DNA-to-RNA cross-attention**: Transcriptional control

![CDT-II Architecture](figures/main/fig1_CDTv2_architecture.png)

**Figure 1.** CDT-II architecture and interpretable attention maps. The model processes genomic DNA (via Enformer) and per-cell RNA expression, producing attention maps that directly correspond to biological relationships.

## Key Results

| Metric | Value |
|--------|-------|
| Overall validation r | 0.64 |
| Per-gene mean r | 0.84 |
| GFI1B network enrichment | 6.6× (P=3.5×10⁻¹⁷) |
| RNA processing module | P=1×10⁻¹⁶ |
| Gradient-based attribution | r=0.83 |

## Paper

- **Preprint**: [bioRxiv](https://www.biorxiv.org/) (coming soon)
- **CDT v1**: [arXiv:2601.01089](https://arxiv.org/abs/2601.01089)

## Repository Structure

```
CDT2/
├── docs/                   # Paper manuscript (.tex, .pdf)
├── figures/main/           # All figures
└── notebooks/
    ├── training/           # Model training
    │   └── CDT_Morris_CRISPRi_CellLevel_Training.ipynb
    └── analysis/           # Results analysis
        ├── CDT_Morris_Prediction_Analysis.ipynb
        ├── CDT_Morris_Attention_Analysis.ipynb
        └── CDT_Morris_Network_Discovery.ipynb
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `CellLevel_Training` | CDT-II model training on Morris STING-seq data |
| `Prediction_Analysis` | Per-gene prediction performance (r=0.84) |
| `Attention_Analysis` | GFI1B regulatory network from attention maps |
| `Network_Discovery` | Convergent RNA processing module discovery |

## Data

Data files are hosted on Hugging Face:

**[nobusama17/CDT2-data](https://huggingface.co/datasets/nobusama17/CDT2-data)**

| File | Description | Size |
|------|-------------|------|
| `morris_celllevel_effects_2361.h5` | Cell-level perturbation effects (TSS) | 41 MB |
| `morris_snp_celllevel_effects_2361.h5` | Cell-level perturbation effects (SNP) | 34 MB |
| `k562_gene_embeddings_aligned.h5` | Gene embeddings from scGPT | 4.4 MB |
| `cdt_morris_celllevel_best.pt` | Trained model weights | 80 MB |
| `morris_28genes_enformer.h5` | Enformer embeddings for 28 TSS genes | 277 MB |
| `morris_snp_enformer.h5` | Enformer embeddings for SNP loci | 4.8 GB |

### Quick Start

The analysis notebooks automatically download data from Hugging Face when running locally:

```python
# Notebooks detect environment and download data automatically
# On Colab: uses Google Drive
# Locally: downloads from Hugging Face

from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="nobusama17/CDT2-data",
    filename="cdt_morris_celllevel_best.pt",
    repo_type="dataset"
)
```

All files including Enformer embeddings are available on Hugging Face and will be downloaded automatically by the notebooks.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- huggingface_hub
- Google Colab (recommended for training and Enformer embedding generation)

## Citation

```bibtex
@article{ota2025cdtii,
  title={Central Dogma Transformer II: An AI Microscope for Understanding Cellular Regulatory Mechanisms},
  author={Ota, Nobuyuki},
  journal={bioRxiv},
  year={2025}
}
```

## License

MIT License

## Author

Nobuyuki Ota
Independent Researcher, Burlingame, CA, USA
nobuyuki.ohta@gmail.com
