
# 🧠 SleepDisorderPredictor

**A machine learning system for predicting sleep disorders using lifestyle metrics collected from wearable devices.**

This project was developed as a final project for CS5100 at Northeastern University by a team of three students. The system processes health-related data, applies feature engineering, trains multiple classifiers, and generates visual analytics and evaluations.

## 📌 Project Highlights

- Complete end-to-end ML pipeline: from raw data to model evaluation
- Uses real-life lifestyle metrics (e.g., screen time, alcohol, sleep duration)
- Supports multiple classification models: SVM, Neural Network, Random Forest
- Includes hyperparameter tuning, EDA visualization, and confusion matrices
- Fully modular Python codebase with clean separation of concerns

## 🔁 Pipeline Overview

1. `data_preprocessing.py` — clean, encode, normalize data
2. `feature_engineering.py` — create new features and transform inputs
3. `train_model.py` — train RF, SVM, and NN models
4. `hyperparameter_tuning.py` — optional: use GridSearchCV for best params
5. `model_evaluation.py` — evaluate models using metrics and visualization
6. `photo.py` — generate distribution and correlation plots
7. `run_all.py` — centralized script to execute the complete pipeline

## 📁 File Structure

SleepDisorderPredictor/
├── data.csv
├── run_all.py
├── data_preprocessing.py
├── feature_engineering.py
├── train_model.py
├── hyperparameter_tuning.py
├── model_evaluation.py
├── photo.py
├── confusion_matrix.png
├── nn_confusion_matrix.png
├── rf_confusion_matrix.png
├── svm_confusion_matrix.png
├── ROC curve.png
├── Training Accuracy and loss.png
├── correlation_heatmap.png
├── sleep_duration_boxplot.png
├── age_distribution.png
├── CS5100 Final Report.pdf
├── requirements.txt
├── .gitignore
└── README.md

## 📊 Key Results

| Model         | Accuracy | F1 Score | Notes                          |
|---------------|----------|----------|-------------------------------|
| Random Forest | 87.5%    | 86.9%    | Best overall performance      |
| Neural Net    | 85.3%    | 84.5%    | Balanced but slower to train |
| SVM           | 82.1%    | 80.7%    | Simpler but lower recall     |

## 📈 Visual Outputs

- confusion_matrix.png: Combined matrix for all models
- ROC curve.png: AUC comparison across models
- correlation_heatmap.png: Feature correlations
- Training Accuracy and loss.png: NN training performance
- sleep_duration_boxplot.png: Sleep patterns by category
- age_distribution.png: Histogram of participant ages

## 🧠 Dataset Description

Your dataset includes anonymized lifestyle and physiological data. Features include:
- Physical activity level
- Sleep duration
- Stress level
- Screen time
- Caffeine and alcohol use
- Age, gender, and environmental metrics

> Note: `data.csv` included here is a sample. Original dataset was preprocessed for training.

## ▶️ How to Run

```bash
pip install -r requirements.txt
python run_all.py
```

## 👩‍💻 Contributions

> Project developed by Chengyi Li, Songkun Li, and **Zhanpeng Tong**

Zhanpeng Tong led all major code development, including:
- Data preprocessing and pipeline design
- Model training scripts (SVM, RF, NN)
- Evaluation metric computation and visualizations
- End-to-end orchestration via `run_all.py`

## 📘 Academic Report

See CS5100 Final Report.pdf for detailed analysis and academic documentation.

## 📄 License

For educational and portfolio use only.  
Feel free to reference this project with attribution.
