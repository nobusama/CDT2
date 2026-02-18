# Central Dogma Transformer II (CDT-II)

**An AI Microscope for Understanding Cellular Regulatory Mechanisms**

[![arXiv](https://img.shields.io/badge/arXiv-2602.08751-b31b1b.svg)](https://arxiv.org/abs/2602.08751)

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

- **CDT-II**: [arXiv:2602.08751](https://arxiv.org/abs/2602.08751)
- **CDT v1**: [arXiv:2601.01089](https://arxiv.org/abs/2601.01089)

## Repository Structure

```
CDT2/
├── data/                   # TSS coordinates and metadata
├── docs/                   # Paper manuscript (.tex, .pdf)
├── figures/main/           # All figures
└── notebooks/
    ├── embeddings/         # Embedding generation
    │   └── Morris_28genes_Enformer.ipynb
    ├── training/           # Model training
    │   └── CDT_Morris_CRISPRi_CellLevel_Training.ipynb
    └── analysis/           # Results analysis
        ├── CDT_Morris_Prediction_Analysis.ipynb
        ├── CDT_Morris_Attention_Analysis.ipynb
        ├── fig2_ablation_study.ipynb
        ├── fig4a_dna_self_attention.ipynb
        ├── fig5b_go_enrichment.ipynb
        └── fig6_encode_validation.ipynb
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Morris_28genes_Enformer` | Enformer embedding generation for 28 target gene TSSs |
| `CellLevel_Training` | CDT-II model training on Morris STING-seq data |
| `Prediction_Analysis` | Per-gene prediction performance (r=0.84) |
| `Attention_Analysis` | GFI1B regulatory network from attention maps |
| `fig2_ablation_study` | Ablation study (Fig 2) |
| `fig4a_dna_self_attention` | DNA self-attention analysis (Fig 4A) |
| `fig5b_go_enrichment` | GO enrichment comparison (Fig 5B) |
| `fig6_encode_validation` | ENCODE validation (Fig 6) |

## Using CDT-II with Your Own Data

CDT-II can be applied to any CRISPRi screen with single-cell RNA-seq readout.

### What You Need

1. **CRISPRi perturbation data** — Single-cell RNA-seq with guide assignments (e.g., STING-seq, Perturb-seq, CROP-seq)
2. **Target gene TSS coordinates** — Chromosome and position (hg38)
3. **Google Colab with GPU** (recommended) or a local GPU

### Pipeline

| Step | Notebook | What It Does |
|------|----------|-------------|
| 1 | `embeddings/Morris_28genes_Enformer.ipynb` | Generate Enformer embeddings for your target gene TSSs |
| 2 | `training/CDT_Morris_CRISPRi_CellLevel_Training.ipynb` | Train CDT-II on your perturbation data |
| 3 | `analysis/CDT_Morris_Prediction_Analysis.ipynb` | Evaluate prediction accuracy per gene |
| 4 | `analysis/CDT_Morris_Attention_Analysis.ipynb` | Extract and interpret attention maps |

### Tips

- Start with the executed notebooks to understand expected outputs
- Gene filtering by cross-dataset reproducibility improves results (see ablation study: 2,361 filtered genes vs 9,335 unfiltered)
- Enformer embeddings are cell-type-agnostic — you only need to regenerate them if your target genes differ

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

## Future Direction

CDT-III is in early planning. Two key directions are being explored:

- **Epigenomic integration** — Incorporating chromatin accessibility, histone modifications, and DNA methylation into the Central Dogma framework, enabling the model to capture the regulatory layers between genome and transcriptome.
- **Drug discovery applications** — Extending CDT's interpretable multi-modal architecture toward predicting and understanding drug responses at the cellular level.

## Citation

```bibtex
@article{ota2026cdtii,
  title={Central Dogma Transformer II: An AI Microscope for Understanding Cellular Regulatory Mechanisms},
  author={Ota, Nobuyuki},
  journal={arXiv preprint arXiv:2602.08751},
  year={2026}
}
```

## License

MIT License

## Author

Nobuyuki Ota
Independent Researcher, Burlingame, CA, USA
nobuyuki.ohta@gmail.com
