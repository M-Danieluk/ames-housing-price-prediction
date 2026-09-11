#  Ames Housing Price Prediction: A Business-Driven Machine Learning Approach

## Project Overview
This project builds an end-to-end Machine Learning pipeline to predict house prices in Ames, Iowa. Instead of stopping at purely mathematical metrics, the analysis translates the final model's results into actionable business insights for real estate investors, flippers, and homeowners.

##  Key Business Insights
Our XGBoost model identified that out of hundreds of property features, the market is overwhelmingly driven by three core factors:
* **Overall Quality (`OverallQual`):** The foundational material and finish quality of the house is the ultimate price driver.
* **Garage Capacity (`GarageCars`):** A highly significant attribute. From a logical standpoint, this acts as a strong proxy for wealth—buyers of premium properties demand multi-car garages.
* **Above-Ground Living Area (`GrLivArea`):** The core square footage, explicitly excluding basement space.

![Top 10 Feature Importances](Ames%20Housing%20Price%20Prediction/images/feature_importance.png)

##  Tech Stack & Methodology
* **Core Stack:** Python, Pandas, Scikit-Learn, XGBoost, Seaborn.
* **Architecture Highlights:** 
  * Strict anti-leakage workflow using `sklearn.pipeline.Pipeline`.
  * Robust hyperparameter tuning via `GridSearchCV` (5-fold Cross-Validation).
  * Direct model comparison: Regularized Linear Models (ElasticNet) vs. Tree-Based Ensembles (XGBoost).

##  How to Reproduce
1. Clone this repository.
2. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt
