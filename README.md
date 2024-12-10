Ethica Labs - AI Governance Dashboard

A user-friendly Streamlit application designed to assess and improve AI governance through model evaluation, fairness analysis, and explainability. This dashboard supports multiple machine learning models and provides insights into classification performance, feature importance, fairness metrics, and instance-level explanations using LIME.

Features
Model Evaluation: Evaluate models like Decision Tree, Random Forest, Logistic Regression, and SVM.
Fairness Analysis: Measure fairness across sensitive features using Fairlearn metrics.
Explainability: Visualize instance-level predictions with LIME explanations.
Interactive UI: Upload datasets, toggle features, and explore results with an intuitive interface.

How to Use
Clone the repository.
download the load_data csv file, as the features are hardcoded for fairness and explainability
Install dependencies with pip install -r requirements.txt.
Run the app: streamlit run app.py.
Upload a dataset and explore model insights.
