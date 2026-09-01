# A Multimodal Explainable Deep Learning Framework for Alzheimer's Disease Diagnosis using 3D MRI and Clinical Data

This project implements a multimodal deep learning framework for Alzheimer's Disease diagnosis, combining 3D Magnetic Resonance Imaging (MRI) with clinical data. The framework includes unimodal (vision-only and clinical-only) and multimodal models, along with explainable AI (XAI) analysis of model predictions.

The framework is developed and evaluated using two publicly available neuroimaging cohorts: the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** and the **Open Access Series of Imaging Studies-3 (OASIS-3)**.

## Data Sources

This study uses two publicly available neuroimaging datasets:

- **ADNI**: [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)
- **OASIS-3**: [https://sites.wustl.edu/oasisbrains/home/oasis-3/](https://sites.wustl.edu/oasisbrains/home/oasis-3/)

Access to both datasets requires registration and approval in accordance with each repository's data use agreement. This repository does not redistribute any raw data.

## Requirements

- Python **3.12.13**

## Installation

1. Create and activate a conda environment:

   ```bash
   conda create -n alzdx python=3.12.13
   conda activate alzdx
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running the pipeline, update the relevant paths and parameters in `config.py` to match your local environment and dataset locations.

## Usage

### 1. Data Preparation

- Clean the dataset using `Preprocessing.ipynb`.
- Curate the MRI images and generate the image mapping file:

  ```bash
  python curate_images.py
  ```

- Build the multimodal dataset (imaging + clinical data):

  ```bash
  python processor_robust.py
  ```

### 2. Model Training

Train each model independently as needed:

| Model | Script |
|---|---|
| Vision (imaging-only) | `python vision_trainer.py` |
| Clinical (tabular-only) | `python tabular_trainer.py` |
| Multimodal (imaging + clinical) | `python robust_multimodal.py` |

Alternatively, run the full set of experimental configurations at once using the provided shell script:

```bash
bash runner.sh
```

### 3. Explainable AI (XAI)

Generate explainability results for each cohort:

- **ADNI**: run `XAI.ipynb`
- **OASIS-3**: run `OASIS3.ipynb`

## Project Pipeline Summary

```
Preprocessing Preprocessing.ipynb → curate_images.py → processor_robust.py
        → vision_trainer.py / tabular_trainer.py / robust_multimodal.py
        → runner.sh (all experiments)
        → XAI.ipynb (ADNI) / OASIS3.ipynb (OASIS-3)
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
