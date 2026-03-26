# Uncertainty-Aware Parking Prediction for Real-Time Contextual Intelligence

This repository contains the codebase for a PhD research project on uncertainty-aware parking occupancy prediction using Bayesian neural networks and neuro-symbolic integration. The work utilises Melbourne on-street parking data, weather data, temporal context, and parking restriction information to support context-aware prediction under uncertainty.

The repository includes baseline classification models, Bayesian models with threshold-based selective prediction, loosely coupled neuro-symbolic methods, and a tightly coupled Bayesian neuro-symbolic model.

---

## 📌 Overview

This project investigates how uncertainty can be explicitly modelled and utilised in real-time prediction systems through:

- Bayesian Neural Networks (BNNs)
- Neuro-symbolic reasoning
- Knowledge-infused learning

The framework is evaluated on real-world parking sensor data from Melbourne, Australia.

---

## 📚 PhD Publications

This repository supports the following publications produced as part of the PhD research:

- Nezhadettehad, A., Zaslavsky, A., Rakib, A., Shaikh, S. A., Loke, S. W., Huang, G. L., & Hassani, A. (2025).  
  *Predicting next useful location with context-awareness: The state-of-the-art.*  
  ACM Transactions on Intelligent Systems and Technology, 16(5), 1–35.  
  https://dl.acm.org/doi/10.1145/3744653  

- Nezhadettehad, A., Zaslavsky, A., Rakib, A., & Loke, S. W. (2025).  
  *Uncertainty-aware parking prediction using Bayesian neural networks.*  
  Sensors, 25(11), 3463.  
  https://www.mdpi.com/1424-8220/25/11/3463  

- Nezhadettehad, A., Zaslavsky, A., Rakib, A., & Loke, S. W. (2025).  
  *Bayesian-Symbolic Integration for Uncertainty-Aware Parking Prediction.*  
  Proceedings of the IEEE Intelligent Transportation Systems Conference (ITSC 2025), Gold Coast, Australia.  
  (Accepted and presented; proceedings forthcoming)  
  https://its.papercept.net/conferences/scripts/abstract.pl?ConfID=91&Number=396  

- Nezhadettehad, A., Zaslavsky, A., Rakib, A., & Loke, S. W.  
  *Tightly Coupled Bayesian Neuro-Symbolic Integration for Uncertainty-Aware Context Prediction.*  
  Manuscript in preparation for journal submission.

---

## 📂 Datasets

The experiments in this repository utilise publicly available datasets from the City of Melbourne and external weather sources:

- **On-street Car Parking Sensor Data – 2019**  
  https://data.melbourne.vic.gov.au/explore/dataset/on-street-car-parking-sensor-data-2019/information/  
  This dataset provides bay-level parking sensor records including timestamps, durations, and location metadata.

- **On-street Car Park Bay Restrictions**  
  https://data.melbourne.vic.gov.au/explore/dataset/on-street-car-park-bay-restrictions/information/  
  This dataset contains structured parking restriction rules such as time windows, duration limits, exemptions, and conditions. It is used as symbolic knowledge in the neuro-symbolic framework.

- **Weather Data**  
  Historical weather data were collected via web crawling from:  
  https://www.wunderground.com/  

---

## 🔗 Notes on Data Usage

- Parking data are aggregated from **bay-level to street-segment level** to align with real-world parking guidance systems.  
- Restriction data are transformed into **structured symbolic features** for both:
  - Loosely coupled reasoning (Phase 2)
  - Tightly coupled learning (Phase 3)
- Weather data are temporally aligned with parking observations to enrich contextual modelling.

---

## 🧠 Research Phases

### Phase 1: Bayesian Neural Networks

- Implements BNNs using variational inference
- Captures epistemic and aleatoric uncertainty
- Uses stochastic forward passes to estimate predictive distributions
- Introduces **threshold-based selective prediction**
  - BNN-20%
  - BNN-30%

---

### Phase 2: Loosely Coupled Neuro-Symbolic Integration

Combines BNN predictions with symbolic reasoning.

**Methods:**
- Decision tree rule extraction
- PLP-style reasoning

**Strategies:**

- **Fallback Reasoning**  
  If BNN confidence is below threshold, symbolic reasoning is used

- **Contextual Refinement**  
  Symbolic reasoning constrains valid classes and refines BNN predictions

---

### Phase 3: Tightly Coupled Bayesian Neuro-Symbolic Model

- Symbolic knowledge is integrated directly into:
  - Input representation
  - Model learning process
- Includes knowledge-aware features and constraints
- Evaluated using accepted and rejected prediction subsets
- Designed for robustness under:
  - Data scarcity
  - Noisy environments

---

## 📁 Repository Structure

```text
.
├── config/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
├── legacy/
│   └── regression/
├── logs/
├── models/
├── papers/
├── src/
│   ├── data_collection/
│   ├── data_processing/
│   ├── models/
│   ├── neuro_symbolic/
│   └── utils/
└── README.md
```

---

## ⚙️ Data Preparation Pipeline

The data preparation process consists of:

1. Creating raw parking event tables
2. Cleaning and structuring data
3. Generating bay-level occupancy time slots
4. Aggregating to street-segment level
5. Processing parking restriction data
6. Merging restriction and occupancy data
7. Generating classification-ready datasets

**Main files:**
```
src/data_processing/create_database.py
src/data_processing/data_cleaning.py
src/data_processing/data_preparation.py
src/data_processing/restriction_preparation.py
src/data_processing/merge_restrictions_with_segments.py
```

---

## 🤖 Classification Models

Implemented models:

- Decision Tree
- Random Forest
- SVM
- RNN / LSTM
- Bayesian Neural Network

**Main files:**
```
src/models/decision_tree_classification.py
src/models/random_forest_classification.py
src/models/svm_classification.py
src/models/rnn_classification.py
src/models/bnn_classification.py
```

---

## 🔗 Neuro-Symbolic Models

### Loosely Coupled Methods

- Rule extraction from decision trees
- PLP-style inference
- Fallback reasoning
- Contextual refinement

**Main files:**
```
src/models/rule_extraction.py
src/neuro_symbolic/plp_inference.py
src/neuro_symbolic/fallback_reasoning.py
src/neuro_symbolic/contextual_refinement.py
```

---

## 🔬 Tightly Coupled Bayesian Neuro-Symbolic Model

- Knowledge-aware feature integration
- Joint learning of data and symbolic constraints
- Selective prediction with uncertainty handling

**Main files:**
```
src/models/tightly_coupled_bnn.py
src/models/evaluate_tightly_coupled.py
```

---

## 🛠 Utility

Common utility for dataset generation:

```
src/utils/util_classification.py
```

Supports:
- Classification dataset generation
- Optional restriction-aware features
- Gaussian noise injection
- Multi-horizon prediction
- 2D / 3D inputs

---

## 🗂 Expected Folder Usage

- `data/raw/` → raw datasets  
- `data/external/` → weather data  
- `data/processed/` → processed datasets  
- `models/` → trained models  
- `logs/` → experiment logs  

---

## ⚙️ Configuration

Located in `config/`:

- `db_config.py` → database credentials  
- `occupancy_classes.py` → class definitions  

---

## ▶️ Run Order

1. **Data preparation**
   - create_database.py  
   - data_cleaning.py  
   - data_preparation.py  

2. **Baseline models**
   - decision_tree_classification.py  
   - random_forest_classification.py  
   - svm_classification.py  
   - rnn_classification.py  
   - bnn_classification.py  

3. **Neuro-symbolic methods**
   - rule_extraction.py  
   - plp_inference.py  
   - fallback_reasoning.py  
   - contextual_refinement.py  

4. **Tightly coupled model**
   - tightly_coupled_bnn.py  
   - evaluate_tightly_coupled.py  

---

## 🌪 Noise Experiments

Supports Gaussian noise injection into:
- Input features  
- Target labels  

Controlled via arguments in dataset generation utilities.

---

## 🎯 BNN Thresholding

- Uses multiple stochastic forward passes  
- Median probability used for prediction  
- Threshold-based selective prediction  
- Low-confidence predictions can:
  - be rejected  
  - be passed to symbolic reasoning  

---

## 📌 Notes

- This repository is designed for **research reproducibility**
- Reflects the full pipeline used in the PhD thesis:
  - Data preparation  
  - Baseline modelling  
  - Bayesian uncertainty modelling  
  - Neuro-symbolic reasoning  
  - Knowledge-infused learning  

---