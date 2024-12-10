import streamlit as st
import pandas as pd
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from fairlearn.metrics import MetricFrame
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load your dataset
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Function to prepare the data
def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Encode categorical variables
    categorical_cols = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    label_encoders = {col: LabelEncoder() for col in categorical_cols}
    
    for col in categorical_cols:
        data[col] = label_encoders[col].fit_transform(data[col])
    
    # Convert target column to integer
    data['loan_status'] = data['loan_status'].astype(int)
    
    # Split data into features (X) and target (y)
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns

# Function to train different models
def train_model(model_type, X_train, y_train):
    if model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_type == "SVM":
        model = SVC(probability=True, random_state=42)  # SVM requires probability=True for LIME
    
    model.fit(X_train, y_train)
    return model

# Fairness Metrics Function
def get_fairness_metrics(X_test, y_test, model, sensitive_feature):
    y_pred = model.predict(X_test)
    metric_frame = MetricFrame(
        metrics=accuracy_score,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test[sensitive_feature]
    )
    return metric_frame

# Streamlit app
st.title('Ethica Labs - AI Governance Dashboard')  # Change title to your company name

# Separator after the title for layout
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar for user inputs (renamed as Ethica Labs)
st.sidebar.markdown("""
    <h1 style='text-align: center; font-size: 36px; font-weight: bold; color: #3b9bfb;'>Ethica Labs</h1>
    """, unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your model dataset", type=["csv", "xlsx"])

# Toggle Sections with Colors and Icons
show_model_evaluation = st.sidebar.checkbox("Show Model Evaluation", value=True)
model_choice = None

if show_model_evaluation:
    model_choice = st.sidebar.selectbox(
        "Select Model for Evaluation",
        ["Decision Tree", "Random Forest", "Logistic Regression", "SVM"]
    )

show_classification_report = st.sidebar.checkbox("Show Classification Report", value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
show_fairness = st.sidebar.checkbox("Show Fairness Metrics", value=True)
if show_fairness:
    sensitive_feature = st.sidebar.selectbox(
        "Select a sensitive feature for fairness assessment:",
        ["person_gender", "person_home_ownership", "person_education", "loan_intent", "previous_loan_defaults_on_file"]
    )

show_explanation = st.sidebar.checkbox("Show LIME Explanation", value=True)

# Apply custom styling for modern UI
def style_dashboard():
    st.markdown("""
    <style>
    /* Background Color */
    .main { 
        background-color: #f4f6fc; 
    }
    .sidebar .sidebar-content {
        background-color: #3b9bfb;
        color: white;
    }
    /* Custom Header */
    h1, h2, h3, h4 {
        color: #3b9bfb;  /* Sky blue */
        font-weight: bold;
        font-size: 32px;
    }
    /* Buttons and Inputs */
    .stButton>button {
        background-color: #3b9bfb;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput input {
        background-color: #e1f5fe;
        border-radius: 8px;
    }
    /* Dataframe */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Hover effects */
    .stButton>button:hover {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the UI styling
style_dashboard()

# Load and preprocess the data
if uploaded_file:
    # Load the data
    data = load_data(uploaded_file)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data)

    # Check if model evaluation toggle is enabled
    if show_model_evaluation and model_choice:
        # Train the selected model
        model = train_model(model_choice, X_train, y_train)
        
        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Show Model Evaluation if toggled on
        st.subheader(f"{model_choice} - Model Evaluation")
        st.markdown(f"**Accuracy**: {accuracy * 100:.2f}%")
        
        # Show Classification Report if toggled on
        if show_classification_report:
            st.subheader(f"{model_choice} - Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
        
        # Show Feature Importance if toggled on (only for tree-based models like Decision Tree and Random Forest)
        if show_feature_importance and model_choice in ["Decision Tree", "Random Forest"]:
            st.subheader(f"{model_choice} - Feature Importance")
            feature_importance = model.feature_importances_
            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            })
            feature_df = feature_df.sort_values(by='Importance', ascending=False)
            fig = px.bar(feature_df, x='Feature', y='Importance', title='Feature Importance', color='Importance', template="plotly_dark")
            st.plotly_chart(fig)
        
        # Show Fairness Metrics if toggled on
        if show_fairness:
            fairness_metrics = get_fairness_metrics(X_test, y_test, model, sensitive_feature)
            st.subheader(f"{model_choice} - Fairness Metrics for {sensitive_feature}")
            st.write(fairness_metrics.by_group)
            
            # Fairness visualization
            fairness_data = fairness_metrics.by_group.reset_index()
            fig = px.bar(fairness_data, x=sensitive_feature, y='accuracy_score', title=f"Loan Approval Rate by {sensitive_feature}", template="plotly_dark")
            st.plotly_chart(fig)

        # Show LIME Explanation if toggled on
        if show_explanation:
            explainer = LimeTabularExplainer(
                training_data=X_train.values,
                training_labels=y_train.values,
                mode='classification',
                feature_names=feature_names
            )
            lime_explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)

            st.subheader(f"{model_choice} - LIME Explanation for the First Instance (Text-based)")
            st.write(lime_explanation.as_list())

            st.subheader(f"{model_choice} - LIME Explanation (Graphical)")
            fig = lime_explanation.as_pyplot_figure()
            st.pyplot(fig)
    else:
        st.warning("Please enable the 'Show Model Evaluation' checkbox and select a model first.")
