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
├── cdt/                    # Model code
│   ├── models/             # CDT-II architecture
│   ├── data/               # Data loading utilities
│   ├── training/           # Training scripts
│   └── interpretability/   # Attention extraction
├── docs/                   # Paper manuscript
├── figures/                # Figures
└── notebooks/              # Analysis notebooks
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

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
