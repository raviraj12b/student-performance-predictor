# 🎓 Student Performance Predictor

> An end-to-end machine learning web application that predicts a student's average exam score based on demographic and preparation factors — built with Python, Scikit-learn, and Flask.

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

<br>


[App Screenshot](data/processed/Screenshot.png)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [ML Pipeline](#ml-pipeline)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Tech Stack](#tech-stack)
- [Key Insights from EDA](#key-insights-from-eda)
- [Author](#author)

---

## Overview

This project predicts a student's average exam score (out of 100) using 5 demographic and preparation features:

- Gender
- Race / Ethnicity
- Parental level of education
- Lunch type (socioeconomic indicator)
- Test preparation course completion

The app walks through the complete ML workflow — data exploration, preprocessing, model training, evaluation, and deployment — making it a fully production-structured machine learning project.

**Dataset:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
1,000 student records · No missing values · 8 original features

---

## Live Demo

🔗 **[Click here to try the app](https://your-deployment-link.onrender.com)**

> Replace the link above after deploying to Render or Railway.

---

## Features

- **Dark dashboard UI** — professional two-column layout with real-time score arc gauge
- **Grade classification** — A / B / C / D with contextual insight message
- **Subject score breakdown** — animated progress bars for Math, Reading, Writing estimates
- **Input validation** — server-side protection against invalid or missing form values
- **Structured logging** — timestamped logs written to console and `logs/app.log`
- **Standalone training script** — retrain the model anytime with `python train.py`
- **Centralized config** — all paths, thresholds, and valid values in `config.py`
- **Fully responsive** — works on desktop and mobile

---

## ML Pipeline

```
Raw CSV
   │
   ├── EDA (Pandas, Matplotlib, Seaborn)
   │     └── Score distributions, gender analysis,
   │         parental education impact, correlation heatmap
   │
   ├── Feature Engineering
   │     └── average_score = (math + reading + writing) / 3
   │
   ├── Preprocessing (ColumnTransformer)
   │     └── OneHotEncoder → 5 columns become 17 binary columns
   │
   ├── Train / Test Split  (80% / 20%, random_state=42)
   │
   ├── Model Comparison
   │     ├── Linear Regression
   │     ├── Decision Tree
   │     ├── Random Forest
   │     └── Gradient Boosting  ← selected
   │
   ├── Evaluation (R², MAE, RMSE)
   │
   └── Saved Pipeline (models/model.pkl)
              │
              └── Flask Web App → Real-time predictions
```

---

## Model Performance

Four regression models were trained and evaluated on the same 80/20 split:

| Model                  |R² Score | MAE     | RMSE     | 
|------------------------|---------|---------|----------|
| **Linear Regression**  | 0.1622  | 10.4902 | 13.4016  |
| Decision Tree          |-0.0753  | 11.8081 |  15.1827 |
| Random Forest          | -0.0252 | 11.4884 | 14.8248  |
| Gradient Boosting      | 0.0858  | 10.832  | 13.9993  |

** Linear Regression** was selected as the best model with:
- **R² =  0.1622** — explains 80.84% of variance in student scores
- **MAE =10.4902** — predictions are off by ~3.5 points on average (0–100 scale)

> Update the table above with your actual metric values after running `train.py`

---

## Project Structure

```
student-performance-predictor/
│
├── app.py                  ← Flask web app (routes, validation, prediction)
├── train.py                ← Standalone model training script
├── config.py               ← All constants: paths, thresholds, valid inputs
├── logger.py               ← Structured logging (console + file)
│
├── notebooks/
│   └── eda_and_training.ipynb   ← Full EDA + model exploration
│
├── data/
│   ├── raw/
│   │   └── StudentsPerformance.csv
│   └── processed/
│       ├── students_cleaned.csv
│       ├── score_distribution.png
│       ├── gender_analysis.png
│       ├── parental_education.png
│       ├── correlation_heatmap.png
│       ├── model_comparison.png
│       └── actual_vs_predicted.png
│
├── models/
│   └── model.pkl           ← Saved best model pipeline
│
├── templates/
│   └── index.html          ← Dark dashboard UI (Jinja2)
│
├── static/
│   └── style.css           ← Dark theme styles
│
├── logs/
│   └── app.log             ← Auto-generated log file
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Git

### 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/student-performance-predictor.git
cd student-performance-predictor
```

### 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Add the dataset
Download [StudentsPerformance.csv](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) from Kaggle and place it in:
```
data/raw/StudentsPerformance.csv
```

### 5 — Train the model
```bash
python train.py
```
This trains all 4 models, selects the best, and saves it to `models/model.pkl`.

### 6 — Run the web app
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## Tech Stack

| Category           | Tools                      |
|--------------------|----------------------------|
| Language           | Python 3.10+               |
| Data manipulation  | Pandas, NumPy              |
| Visualization      | Matplotlib, Seaborn        |
| Machine learning   | Scikit-learn               |
| Web framework      | Flask                      |
| Frontend           | HTML5, CSS3, Jinja2        |
| Version control    | Git, GitHub                |
| Environment        | Virtual environment (venv) |

---

## Key Insights from EDA

1. **No missing values** — dataset is clean with 1,000 complete records
2. **Test preparation course** has a strong positive effect across all 3 subjects
3. **Lunch type** is a significant socioeconomic predictor — standard lunch students score ~8 points higher on average
4. **Parental education** correlates positively with student performance — master's degree parents → highest scoring students
5. **Reading and writing scores** are highly correlated (r ≈ 0.95) — they carry nearly identical information
6. **Score distributions** are approximately normal — well-suited for regression models
7. **Gender gap** — females score higher in reading/writing; males slightly higher in math

---

## What I Learned

- Building an end-to-end ML pipeline from raw CSV to deployed web app
- Using `ColumnTransformer` and `Pipeline` to prevent data leakage
- Comparing multiple regression models using R², MAE, and RMSE
- Structuring a Python project with `config.py`, `logger.py`, and separate training scripts
- Building a Flask app with form handling, server-side validation, and Jinja2 templating
- Writing production-aware code: logging, error handling, input validation

---

## Author

**Your Name**
- GitHub: [@raviraj12b](https://github.com/raviraj12b)
- LinkedIn: [linkedin.com/in/rajeshborkar01/]((https://www.linkedin.com/in/rajeshborkar01/))
- Email: rajeshborkar04@gmail.com

---

## License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ using Python · Scikit-learn · Flask
</p>
