# SIGNAL: Sign Annotation aLgorithm for Protein-Protein Interaction Networks
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

![Figure 1](https://github.com/user-attachments/assets/0cf76fd6-c0d4-4b6b-a8ed-6663156b7d12)

## Overview

SIGNAL (Sign Annotation aLgorithm) is a computational method for annotating protein-protein interaction (PPI) networks with activation/repression signs based on cause-effect data. The algorithm uses network propagation techniques to quantify the influence of each edge on gene expression changes, leveraging the observation that negatively signed edges have greater influence on pathway effects compared to positive ones.

## Features

- **Sign prediction for PPIs**: Annotate protein-protein interactions as activating (+) or repressing (-)
- **Network propagation-based approach**: Leverages information flow through the network
- **High accuracy**: Achieves AUC of 0.98 for kinase/phosphatase interactions
- **Phenotype reconstruction**: Can predict knockout effects and phenotypes
- **Support for multiple organisms**: Currently supports *S. cerevisiae* and *H. sapiens*

## Installation

### Requirements

- Python 3.9 or higher
- Required Python packages:
  ```
  gseapy==1.1.7
  joblib==1.4.2
  lxml==5.4.0
  matplotlib==3.10.3
  networkx==3.4.2
  numpy==2.3.1
  pandas==2.3.0
  Requests==2.32.4
  scikit_learn==1.7.0
  scipy==1.16.0
  statsmodels==0.14.4
  ```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/L-F-S/PPI_Network_Signer.git
cd PPI_Network_Signer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running SIGNAL on S. cerevisiae

1. **Basic usage**: Generate signed PPI network
```bash
python applySIGNAL_server.py -s S_cerevisiae
```

2. **Custom edge list**: Predict signs for specific edges
```bash
python applySIGNAL_server.py -s S_cerevisiae -e path/to/edges.tsv
```

3. **Full pipeline with visualization**:
```bash
python scripts/run_full_pipeline.py -s S_cerevisiae
```

### Input Data Format

SIGNAL requires three main types of input data:

1. **Base PPI Network**: Tab-separated file with columns: source, target
2. **Signed Training Data**: Known signed interactions in `.lbl.tsv` format
3. **Knockout Signatures**: Gene expression changes following gene knockouts

See `input/S_cerevisiae/` for example data formats.

## Repository Structure

```
SIGNAL/
├── applySIGNAL_server.py          # Main script for applying SIGNAL
├── SIGNAL_ft_gen_PARALLEL.py      # Parallel feature generation
├── train_and_vis3_5.py            # Training and visualization
├── glob_vars.py                   # Global configuration variables
├── input/                         # Input data directory
│   ├── S_cerevisiae/             # Yeast data
│   │   ├── network/              # PPI networks
│   │   ├── edges/                # Edge lists
│   │   ├── signatures/           # Knockout signatures
│   │   └── labels/               # Known signed edges
│   └── H_sapiens/                # Human data (same structure)
├── output/                        # Output directory
│   ├── S_cerevisiae/
│   │   ├── features/             # Generated features
│   │   ├── models/               # Trained models
│   │   └── predictions/          # Predicted signs
│   └── H_sapiens/
├── scripts/                       # Utility scripts
├── Validation/ 
    ├──crossvalidations/
    ├──phenotype_reconstruction_model/
        ├──input/
            TLM data input
        ├──output/
        ├──scripts/
| other/
    ├──ANAT_validation_pipeline/
└── functional_enrichment/         # GO enrichment analysis
    ├── data/
    ├── imgs/
    ├── modules/
    ├── output/
```

## Detailed Usage

### 1. Feature Generation

Generate features for edge sign prediction:

```bash
python SIGNAL_ft_gen_PARALLEL.py \
    -s S_cerevisiae \
    -e input/S_cerevisiae/edges/trainedges.tsv \
    -n input/S_cerevisiae/network/S_cerevisiae_patkar_kegg_kpi_ubiq.nx \
    -p input/S_cerevisiae/signatures/
```

### 2. Model Training

Train the random forest classifier:

```bash
python train_and_vis3_5.py \
    -s S_cerevisiae \
    --train \
    --features output/S_cerevisiae/features/traindata_perturb.ft
```

### 3. Sign Prediction

Predict signs for new edges:

```bash
python applySIGNAL_server.py \
    -s S_cerevisiae \
    -e input/S_cerevisiae/edges/test_edges.edges \
    --predict_only
```

### 4. Validation

Run cross-validation:

```bash
python validation/crossvalidation/run_cv.py \
    -s S_cerevisiae \
    -k 5
```

## Output Files

- **Predicted signs**: `output/{species}/predictions/` - Tab-separated files with edge signs
- **Trained models**: `output/{species}/models/` - Pickled Random Forest models
- **Features**: `output/{species}/features/` - Generated feature matrices

## Advanced Features

### Phenotype Reconstruction

Reconstruct telomere length maintenance phenotypes:

```bash
python validation/TLM_phenotype/reconstruct_tlm.py
```

### Functional Enrichment

Perform GO enrichment analysis on predicted negative edges:

```bash
python functional_enrichment/analyze_negative_edges.py \
    -s S_cerevisiae \
    -p output/S_cerevisiae/predictions/all_edges.sign
```

## Parameters

Key parameters can be configured in `glob_vars.py`:

- `ALPHA`: Network propagation smoothing parameter (default: 0.8)
- `EPSILON`: Convergence threshold (default: 0.01)
- `TAU`: Sign assignment threshold (default: 0.5)
- `MIN_PATHS`: Minimum shortest paths for phenotype reconstruction (default: 100)


## Contact

- **Lead Developer**: Lorenzo Federico Signorini

## Acknowledgments

This project was funded by the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 859962.
