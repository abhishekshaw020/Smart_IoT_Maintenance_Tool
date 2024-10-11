import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit app title
st.title('Smart IoT Maintenance Tool')

# File upload
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader('Dataset Preview')
    st.write(df.head())

    # Data Preprocessing
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.drop(['Device_ID'], axis=1, inplace=True)  # Drop Device_ID if present
    df.dropna(inplace=True)  # Remove rows with missing values

    # Display basic statistics
    st.subheader('Basic Statistics of the Dataset')
    st.write(df.describe())

    # Exploratory Data Analysis (EDA)
    st.subheader('Exploratory Data Analysis (EDA)')
    
    # 1. Distribution of Failures
    st.subheader('Distribution of Failures')
    fig, ax = plt.subplots()
    sns.countplot(x='Failure', data=df, ax=ax)
    st.pyplot(fig)

    # 2. Correlation Heatmap
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt)

    # Splitting the data into features and target
    X = df.drop('Failure', axis=1)

    # ** NEW STEP: Ensure only numeric columns are included in X **
    X = X.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns
    y = df['Failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Training: Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Model Evaluation: Predicting and displaying accuracy
    st.subheader('Model Evaluation')
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Classification Report
    st.subheader('Classification Report')
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    st.subheader('Feature Importance')
    importance = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
    st.pyplot(fig)

    # Prediction Section
    st.subheader('Make a Prediction')

    # User input for prediction
    temperature = st.number_input('Temperature', min_value=0.0, max_value=100.0, value=70.0)
    vibration = st.number_input('Vibration', min_value=0.0, max_value=10.0, value=1.0)
    pressure = st.number_input('Pressure', min_value=90.0, max_value=110.0, value=101.3)

    # Creating input DataFrame for the model
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Vibration': [vibration],
        'Pressure': [pressure]
    })

    # Predict button
    if st.button('Predict'):
        prediction = rf_model.predict(input_data)
        if prediction == 1:
            st.write('Prediction: Failure is likely to occur.')
        else:
            st.write('Prediction: No failure is expected.')

    # New Feature: Show Data Distribution
    if st.checkbox('Show Data Distribution'):
        st.subheader('Data Distribution for Features')
        for feature in X.columns:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

    # New Feature: Show User Guidance
    # Sidebar for user guidance
st.sidebar.subheader('User Guidance')
st.sidebar.write("1. Upload your dataset in CSV format.")
st.sidebar.write("2. Review the dataset preview and basic statistics.")
st.sidebar.write("3. Explore EDA plots.")
st.sidebar.write("4. Use the inputs to predict device failure based on sensor readings.")
st.sidebar.write("If you like my project, do follow me on:")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/abhishaw020/)")
st.sidebar.write("[Medium](https://medium.com/@abhishekshaw020)")
st.sidebar.write("[Kaggle](https://www.kaggle.com/abhishekshaw020)")
st.sidebar.write("[GitHub](https://github.com/abhishekshaw020)")
