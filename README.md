# Bengaluru House Price Prediction

This project focuses on predicting house prices in Bengaluru using machine learning techniques. The idea came from personal observation — property prices in Bangalore vary wildly depending on location, size, and amenities, and it’s often difficult to judge whether a price is reasonable or not.

The model takes common housing features and estimates the expected price in lakhs.

---

## Project Overview

Finding a home in Bengaluru can be confusing due to inconsistent pricing across areas. This project aims to bring some clarity by using historical housing data and machine learning models to predict prices based on key property attributes such as location, square footage, and number of rooms.

---

## Dataset

The dataset is sourced from Kaggle and contains **13,320 property listings** from Bengaluru.

### Key columns include:
- Location (1300+ distinct areas)
- Size (BHK format)
- Total square footage
- Number of bathrooms
- Number of balconies
- Area type
- Price (in lakhs)

---

## Work Done

### 1. Data Cleaning
Several preprocessing steps were required to make the dataset usable:
- Handled missing values in `bath`, `balcony`, and `society`
- Converted size values like `"2 BHK"` into numeric form
- Cleaned `total_sqft` values that appeared as ranges (e.g. `"2100-2850"`) by taking their average
- Created new features such as:
  - `BHK`
  - `price_per_sqft`

---

### 2. Outlier Removal
Unrealistic data points were removed to improve model reliability:
- Properties priced above **200 lakhs**
- Houses with less than **300 sqft per bedroom**
- Listings where **bathrooms > BHK + 2**

---

### 3. Models Used

The following regression models were trained and evaluated:

1. **Linear Regression**  
   Used as a baseline model.

2. **Decision Tree Regressor**  
   Captures non-linear relationships in the data.

3. **Random Forest Regressor**  
   Delivered the best performance overall.

---

## Model Performance

| Model | R² Score | MAE | RMSE |
|------|----------|-----|------|
| Linear Regression | 0.78 | 12.5 | 15.01 |
| Decision Tree Regressor | 0.84 | 10.2 | 13.44 |
| **Random Forest Regressor** | **0.89** | **8.7** | **12.08** |

The Random Forest model explains **89% of the variance** in house prices and produced the lowest error values.

---

## Tech Stack

- Python 3.x  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- Jupyter Notebook  

---

## How to Run the Project

1. Clone the repository
2. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
