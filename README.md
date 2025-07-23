# molecular_dissolution_prediction

This repository contains the source code and trained models for the study:

**"Integrating artificial intelligence and physiologically based pharmacokinetic modelling to predict in vitro and in vivo fate of amorphous solid dispersions."**
 
📄 Publication DOI: *[To be added upon acceptance]*

---

## 🌟 Highlights

- Predicts **molecular dissolution profiles** of ASD formulations using machine learning models
- Uses **TabPFN**, a transformer-based foundation model for tabular data
- Includes model interpretability using **SHAP analysis**

---

## 📦 Contents

| Folder/File       | Description                                         |
|-------------------|-----------------------------------------------------|
| `Diss/,AUD/,Dmax` | Pre-trained `.pkl` models                           |
| `src/`            | Scripts for preprocessing, prediction, SHAP         |
| `requirements.txt`| Python dependencies                                 |

---

## 🛠️ Installation

We recommend using a virtual environment.

```bash
git clone https://github.com/xappiny/Molecular_diss.git
cd Molecular_diss
pip install -r requirements.txt
