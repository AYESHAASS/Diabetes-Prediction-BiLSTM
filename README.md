BiLSTM-based Predictive Modeling for Diabetes Mellitus


📌 Project Overview
This repository contains a modularized deep learning pipeline for predicting diabetes risk using medical features. The core of this project explores the performance of Bidirectional Long Short-Term Memory (BiLSTM) networks on structured medical datasets.
While BiLSTMs are traditionally used for sequential data, this research evaluates their ability to capture complex feature relationships in tabular data through high-dimensional mapping and 
aggressive regularization.


🚀 Engineering Highlights
Unlike standard academic scripts, this project is built for robustness and reproducibility:
Leakage-Free Pipeline: Data splitting is performed prior to scaling and resampling (SMOTEENN) to ensure zero data contamination.
Feature Engineering: Implemented Log Transformations for skewed features (Insulin/Pedigree) and created medical risk-factor interactions to improve signal-to-noise ratio.
Advanced Regularization: Utilized L2 Kernel Regularization, Activity Regularization, and Batch Normalization to mitigate the generalization gap inherent in training Deep Learning models on small tabular datasets.
Modular Architecture: Separated logic into preprocessing, model_definition, and execution modules for enterprise-grade maintainability.


📊 Performance Benchmarks
Note: These results represent the final stabilized model after controlling for overfitting.
Metric	      Training Score	Testing Score
Accuracy	      84.08%	        72.08%
Recall	          90.15%	        87.04%
Precision	      81.20%	        58.02%
F1-Score	      85.44%	        69.63%


Research Note: The observation of an ~12% generalization gap is documented as a characteristic of BiLSTM architectures when applied to non-sequential, small-scale tabular data. This project prioritizes Generalization and Honesty over inflated leaked accuracy scores.


🛠 Tech Stack
Deep Learning: TensorFlow 2.x, Keras
Data Science: Pandas, NumPy, Scikit-learn
Resampling: Imbalanced-learn (SMOTEENN)
Environment: Python 3.9+


📂 Repository Structure
code
Text
├── data/               # Contains diabetes.csv
├── src/
│   ├── preprocessing.py # Log transforms, SMOTEENN, and Leakage-free scaling
│   └── model.py         # BiLSTM architecture with L2/Dropout
├── main.py              # Orchestration script with EarlyStopping
├── requirements.txt     # Dependency list
└── README.md


⚙️ Installation & Usage
Clone the repo:
code
Bash
git clone https://github.com/AYESHAASS/Diabetes-Prediction-BiLSTM.git
cd Diabetes-Prediction-BiLSTM


Install Dependencies:
code
Bash
pip install -r requirements.txt
Execute Pipeline:
code
Bash
python main.py
