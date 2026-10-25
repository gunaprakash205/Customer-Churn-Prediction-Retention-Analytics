BankGuard AI — Customer Churn Prediction & Retention Analytics


A machine learning project that analyzes customer data and predicts the likelihood of customer churn. The model is deployed using Streamlit for real-time churn prediction.




# Customer Churn Prediction & Retention Analytics

A machine learning project that predicts whether a bank customer is likely to leave, using customer demographics, account relationships, and transaction activity. This project covers the complete workflow, from data preparation and model evaluation to hyperparameter tuning and a Streamlit web application.

## About the Project

Customer churn is an important challenge for banks. When customers stop using their services, businesses can lose revenue and valuable long-term relationships.

The goal of this project is to identify patterns in customer data that may be associated with attrition. I built and evaluated several classification models, selected Gradient Boosting for further tuning, and developed a Streamlit application where users can enter customer details and view a prediction.

The application predicts one of two classes:

- **Existing Customer** — the customer belongs to the existing-customer class.
- **Attrited Customer** — the customer belongs to the attrited-customer class.

The model's output can support customer-retention analysis, but it should not be treated as a guarantee of future customer behavior.

## Project Objectives

- Understand customer data and explore attrition patterns.
- Prepare numerical and categorical features for machine learning.
- Compare multiple classification algorithms.
- Tune the Gradient Boosting model using cross-validation.
- Evaluate performance using multiple classification metrics.
- Deploy the trained model through a user-friendly Streamlit application.

## Dataset

The project uses the BankChurners dataset, containing **10,127 customer records and 23 columns**.

The dataset includes information such as:

- Customer age, gender, education, and marital status.
- Income category and card category.
- Length of the customer relationship with the bank.
- Number of products held by the customer.
- Recent inactivity and contact frequency.
- Credit limit, revolving balance, and available credit.
- Transaction amounts, transaction counts, and changes in transaction activity.

### Target variable

The target column is `Attrition_Flag`.

| Customer group | Encoded value | Records |
|---|---:|---:|
| Existing Customer | 0 | 8,500 |
| Attrited Customer | 1 | 1,627 |
| **Total** | | **10,127** |

The target classes are imbalanced, meaning that the dataset contains considerably more existing customers than attrited customers. Because of this, accuracy alone is not enough to evaluate the model.

## Data Preparation

The data preparation workflow included:

1. Separating the target variable from the input features.
2. Removing `CLIENTNUM`, because it is a customer identifier rather than a behavioral feature.
3. Excluding the two Naive Bayes classifier output columns included in the dataset to avoid potential information leakage.
4. Splitting the dataset into training and testing sets using an 80:20 ratio.
5. Using stratification to preserve the class distribution across both sets.
6. Filling missing numerical values using median imputation.
7. Scaling numerical features.
8. Filling missing categorical values using the most frequent category.
9. Converting categorical variables into numerical representations using one-hot encoding.

The train-test split used `random_state=42` and `stratify=y`. The test set contains 2,026 records.

Preprocessing and classification were combined in a scikit-learn pipeline so that the same transformations could be applied during training and prediction.

## Models Evaluated

I compared six classification algorithms to understand how they performed on the customer attrition dataset.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.98% | 76.75% | 53.85% | 63.29% | 91.67% |
| Decision Tree | 93.78% | 77.87% | 85.54% | 81.52% | 91.61% |
| Random Forest | 95.46% | 85.85% | 85.85% | 85.85% | 98.46% |
| Extra Trees | 91.91% | 93.51% | 53.23% | 67.84% | 97.05% |
| Gradient Boosting | 96.40% | 95.65% | 81.23% | 87.85% | 98.77% |

Although Hist Gradient Boosting achieved stronger results in the initial comparison, I chose to continue with the standard **Gradient Boosting Classifier** for the final model.

## Hyperparameter Tuning

To improve the standard Gradient Boosting model, I used `GridSearchCV` with five-fold `StratifiedKFold` cross-validation.

The search configuration included:

- Five cross-validation folds.
- Shuffling enabled.
- `random_state=42`.
- Recall as the optimization metric.
- 16 parameter combinations, resulting in 80 model fits.

### Best parameters

```python
{
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "n_estimators": 150
}
```

The best mean cross-validation recall was **86.94%**.

Recall was chosen as the tuning metric because identifying customers who may leave is an important objective in attrition analysis.

## Final Model Results

The tuned Gradient Boosting pipeline achieved the following results on the held-out test set.

| Evaluation metric | Result |
|---|---:|
| Accuracy | 96.79% |
| Precision | 96.76% |
| Recall | 82.77% |
| F1-score | 89.22% |
| ROC-AUC | 99.05% |

### Classification report

| Customer class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Existing Customer | 96.80% | 99.47% | 98.12% | 1,701 |
| Attrited Customer | 96.76% | 82.77% | 89.22% | 325 |
| Macro average | 96.78% | 91.12% | 93.67% | 2,026 |
| Weighted average | 96.79% | 96.79% | 96.69% | 2,026 |

### Understanding the metrics

- **Accuracy:** The percentage of all test records classified correctly.
- **Precision:** How often predictions for a particular class are correct.
- **Recall:** How many actual members of a class the model successfully identifies.
- **F1-score:** A balance between precision and recall.
- **ROC-AUC:** Measures how effectively the model distinguishes between the two classes across different classification thresholds.

For customer churn, recall is especially important because failing to identify a customer who may leave could mean missing a potential retention opportunity. Precision is also important because unnecessary outreach can consume time and resources.

These results reflect the evaluation performed during this project. Performance on future or external data may differ.

## Streamlit Web Application

The project includes a Streamlit application that makes the trained model easier to use without manually running prediction code in a notebook.

### Application features

- A customer information form.
- Inputs for demographic details, account relationships, and transaction activity.
- Prediction of the existing-customer or attrited-customer class.
- Estimated churn probability when supported by the saved model.
- A visual risk indicator based on the estimated probability.
- An option to download the prediction as a CSV file.
- A dark navy and violet dashboard design.

The app uses the saved preprocessing-and-model pipeline to transform the entered information and generate a prediction.

The displayed probability is an estimate produced by the model, not a guarantee. The application's risk categories are illustrative and should not be interpreted as validated business decision thresholds.

## Technology Stack

- **Python** — primary programming language.
- **Pandas** — data manipulation and preparation.
- **scikit-learn** — preprocessing, model training, cross-validation, tuning, and evaluation.
- **Joblib** — saving and loading the trained pipeline.
- **Streamlit** — interactive web application.
- **Jupyter Notebook** — experimentation and model development.

## Project Structure

```text
customer-churn-prediction-retention-analytics/
│
├── app.py
├── customer_chrun.joblib
├── requirements.txt
└── README.md
```

- `app.py` — Streamlit application.
- `customer_chrun.joblib` — saved, fitted preprocessing and classification pipeline.
- `requirements.txt` — Python dependencies.
- `README.md` — project documentation.

**Important:** The application currently expects the model file to be named `customer_chrun.joblib`. Keep this spelling consistent, or update the `MODEL_PATH` variable in `app.py` if you choose a different filename.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/gunaprakash205/customer-churn-prediction-retention-analytics.git
cd customer-churn-prediction-retention-analytics
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

A basic `requirements.txt` contains:

```text
streamlit
pandas
scikit-learn
joblib
```

For reliable deployment, pin scikit-learn to the version used when the model was trained and saved.

### 4. Launch the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `customer_chrun.joblib`, `requirements.txt`, and `README.md` to your GitHub repository.
2. Open Streamlit Community Cloud and create an app connected to your repository.
3. Select `app.py` as the main application file.
4. Deploy the application.
5. Check the deployment logs if an error occurs.
6. Confirm that the model filename and dependency versions match the application configuration.

The model file must be available in the deployed project directory. Do not commit private customer information, credentials, or sensitive data to a public repository.

## Saving the Trained Model

The saved artifact should contain the entire fitted pipeline, including preprocessing and the trained classifier.

For example, in the training notebook:

```python
import joblib

joblib.dump(tuned_model, "customer_chrun.joblib")
```

Here, `tuned_model` must refer to the fitted pipeline used for the final evaluation. Saving only the classifier may cause prediction errors if the app expects the preprocessing steps to be included.

Models saved with Joblib should generally be loaded using a compatible environment, particularly the same scikit-learn version used during training.

## Limitations and Responsible Use

- The model learns from historical data and may not generalize perfectly to future customers.
- Strong test-set results do not guarantee the same performance in production.
- The predicted class and probability should be treated as decision-support information rather than certainty.
- Before using predictions in real retention campaigns, evaluate model performance on representative new data.
- Review privacy, fairness, and applicable organizational requirements before using customer information.
- Predictions should support appropriate human review rather than serve as the sole basis for consequential decisions.

## Future Improvements

Some possible extensions for this project are:

- Add a confusion matrix and feature-importance visualization.
- Evaluate probability calibration.
- Compare classification thresholds based on retention costs and benefits.
- Add batch predictions for uploaded customer CSV files.
- Monitor performance on new data and retrain when needed.
- Use interpretability techniques such as permutation importance or SHAP.

## Author

**Guna Prakash Pokala**

This project demonstrates an end-to-end machine learning workflow, from data preparation and model comparison to hyperparameter tuning, evaluation, and deployment through an interactive web application.

---

*Built with Python, Pandas, scikit-learn, Joblib, and Streamlit.*
