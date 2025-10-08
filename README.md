# Energy Consumption Forecasting using LSTMs ⚡️

A time-series forecasting project that uses an LSTM Recurrent Neural Network to predict future household energy consumption. This repository demonstrates a complete workflow for building and evaluating RNNs for a regression task, from raw data to a predictive model.

![-](https://img.shields.io/badge/Python-3.8+-blue.svg)
![-](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![-](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Project Workflow](#project-workflow)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Learning Outcomes](#learning-outcomes)
- [License](#license)

---

## Project Overview
The goal of this project is to accurately forecast future energy usage based on historical data. By analyzing past consumption, cost, and time-based patterns (like the hour of the day or day of the week), the model learns the complex temporal dependencies in energy demand. This is a practical application of deep learning for time-series analysis.

The model is trained on a multivariate dataset to predict a single future value: the next hour's energy `USAGE`.

---

## Key Features
- **Data Preprocessing**: Cleaning and structuring time-series data using Pandas, including handling timestamps and resampling.
- **Feature Engineering**: Creating insightful features from a datetime index (e.g., hour, day of week, month) to help the model capture seasonality.
- **Chronological Splitting**: Correctly splitting time-series data into training and testing sets without data leakage.
- **Data Scaling**: Applying `MinMaxScaler` to normalize features for optimal neural network performance.
- **LSTM Model Architecture**: Building a stacked LSTM model with Dropout layers in Keras to capture long-range dependencies and prevent overfitting.
- **Model Evaluation**: Measuring performance using standard regression metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

---

## Technology Stack
- **Python 3.8+**
- **TensorFlow / Keras**: For building and training the deep learning model.
- **Pandas**: For data manipulation and time-series analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For data splitting and scaling.
- **Matplotlib / Seaborn**: For data visualization and plotting results.

---

## Project Workflow
The project follows a standard machine learning pipeline:

1.  **Data Loading & Cleaning**: The initial dataset is loaded, timestamps are parsed, and the data is cleaned for analysis.
2.  **Exploratory Data Analysis (EDA)**: The data is visualized to identify trends, seasonality, and potential outliers.
3.  **Feature Engineering**: New time-based features are created from the timestamp index.
4.  **Preprocessing**: The dataset is split chronologically into training and test sets, and all features are scaled to a range of [0, 1].
5.  **Data Windowing**: The time series is transformed into supervised learning samples of `(X, y)` pairs, where `X` is a sequence of past observations and `y` is the target value to predict.
6.  **Model Building**: A `Sequential` Keras model with LSTM layers is defined.
7.  **Training & Evaluation**: The model is trained on the training data and evaluated on the held-out test set to assess its forecasting accuracy.

---

## Setup and Installation

Follow these steps to set up the project environment on your local machine.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

**2. Create and activate a virtual environment (recommended):**
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```
*(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal after installing the libraries listed in the Technology Stack.)*

---

## Usage
To run the project, execute the main script or Jupyter Notebook:

- **If using a Python script (`train.py`):**
  ```bash
  python train.py
  ```
- **If using a Jupyter Notebook (`notebook.ipynb`):**
  ```bash
  jupyter notebook notebook.ipynb
  ```
The script will handle data loading, preprocessing, model training, and will output the final evaluation metrics and a plot of the results.

---

## Results
The model's performance was evaluated on the held-out test set. The metrics below are calculated on the scaled data (values between 0 and 1) and indicate a strong predictive performance.

- **Mean Absolute Error (MAE)**: `0.0205`
- **Mean Squared Error (MSE)**: `0.0024`
- **Root Mean Squared Error (RMSE)**: `0.0488`

---

## Learning Outcomes
This project provides hands-on experience with the key concepts required for time-series forecasting with deep learning:
- **End-to-End Workflow**: Gained experience in managing a complete machine learning project from data ingestion to model evaluation.
- **Temporal Data Handling**: Learned techniques for cleaning, resampling, and engineering features specifically for time-series data.
- **RNN Application**: Understood the practical application of LSTMs for regression and how to structure a model in Keras.
- **Time-Series Preprocessing**: Mastered the crucial concepts of data windowing and the importance of chronological (non-shuffled) data splitting.
- **Forecasting Evaluation**: Became familiar with interpreting regression metrics (MAE, RMSE) in the context of a forecasting problem.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
