# Customer Churn Prediction

An end-to-end data science project that predicts which telecom customers are likely to leave, explains why, and serves predictions through a Streamlit web app.

## Business Problem

Acquiring a new customer costs far more than keeping an existing one. If a telecom company can spot customers who are likely to churn, it can offer them a retention deal before they leave. This project builds a model that scores each customer's churn risk and identifies the factors driving it.

## Dataset

- **Source:** Telco Customer Churn (IBM sample data, available on Kaggle)
- **Size:** 7,043 customers, 21 columns (7,032 rows after cleaning)
- **Target:** `Churn` (Yes/No), with an overall churn rate of about 26.6%
- **Features:** demographics, account details (contract, tenure, payment method), services (internet, phone, add-ons) and charges

## Approach

1. **Cleaning:** converted `TotalCharges` to numeric and removed 11 blank rows. Encoded `Churn` as 0/1.
2. **EDA:** compared churn rates across contract type, internet service, payment method, tenure and charges.
3. **Feature engineering:** one-hot encoding, a stratified 80/20 train/test split, and scaling of numeric columns (fit on training data only to avoid leakage).
4. **Class imbalance:** only about 27% of customers churn, so I used class weights (`class_weight="balanced"` and `scale_pos_weight`) instead of resampling.
5. **Modelling:** compared Logistic Regression, Random Forest and XGBoost.
6. **Explainability:** logistic regression coefficients and SHAP values for XGBoost.
7. **Deployment:** a Streamlit app that returns a churn risk score for a single customer.

## Model Results

Evaluated on the held-out 20% test set (1,407 customers):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.726 | 0.490 | 0.794 | 0.606 | 0.835 |
| Random Forest | 0.787 | 0.627 | 0.489 | 0.550 | 0.819 |
| XGBoost | 0.729 | 0.493 | 0.783 | 0.605 | 0.834 |

**Why recall matters most here:** missing a customer who is about to leave costs more than sending a retention offer to someone who would have stayed. Random Forest has the best accuracy but catches fewer than half of the churners, so it is the wrong choice for this goal. Logistic Regression and XGBoost both catch about 78-79% of churners. XGBoost is used in the app.

## Key Findings

Churn rate by group (from the data):

| Segment | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Fiber optic internet | 41.9% |
| DSL internet | 19.0% |
| Electronic check payment | 45.3% |
| Senior citizens | 41.7% |

- Churned customers have a median tenure of **10 months**, against **38 months** for customers who stayed.
- SHAP shows the strongest drivers are **contract type, tenure, fiber optic internet, monthly charges and electronic check payment**.
- Customers with **Online Security** and **Tech Support** are less likely to churn.

## Business Recommendations

1. **Move month-to-month customers to longer contracts** with a discount or perk, especially in their first year.
2. **Focus early retention on new customers.** Most churn happens in the first months, so run onboarding and check-in campaigns.
3. **Investigate the fiber optic experience.** Churn is more than double that of DSL, which may point to price or service-quality problems.
4. **Encourage automatic payments** over electronic check.
5. **Promote Online Security and Tech Support add-ons,** which are linked to lower churn.

## Streamlit App

Enter a customer's details and get a churn risk score with a High / Medium / Low message.

| High risk customer | Low risk customer |
|---|---|
| ![High risk](screenshots/high_risk.png) | ![Low risk](screenshots/low_risk.png) |

## Limitations

- Precision is about 49%, so roughly half of the customers flagged as high risk would not actually have churned. This is the trade-off for catching most real churners.
- The score is a risk score, not an exact probability, because the churn class was weighted.
- The data is a single snapshot with no time dimension, so the model cannot track how behaviour changes over time.

## How to Run

```bash
git clone https://github.com/gopiarul/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, run the notebook `churn_analysis.ipynb` from top to bottom.

## Project Structure

```
churn-prediction/
├── churn_analysis.ipynb   # cleaning, EDA, modelling, SHAP
├── app.py                 # Streamlit app
├── churn_model.pkl        # trained XGBoost model
├── scaler.pkl             # fitted StandardScaler
├── columns.pkl            # feature column order
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── plots/                 # EDA and SHAP charts
├── screenshots/           # app screenshots
└── requirements.txt
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Streamlit

## Author

Gopinath · [GitHub](https://github.com/gopiarul)
