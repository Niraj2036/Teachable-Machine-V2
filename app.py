import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def preprocess_data(df, max_unique_values=10):
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # Fill missing values with median for numeric columns
    
    # Handle non-numeric values
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass  # If conversion to float fails, keep the column as it is (assume it's string data)
    
    # Drop unnecessary columns with a high number of unique categorical values
    unnecessary_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].nunique()
            if unique_values > max_unique_values:
                unnecessary_columns.append(col)
    df = df.drop(columns=unnecessary_columns)
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].unique()) <= 2:  # Binary categorical variable
            df[col] = pd.factorize(df[col])[0]
        else:  # Categorical variable with more than 2 categories
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    return df
def evaluate_regressor(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return r2
def evaluate_classifier(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy
def drop_by_importance_threshold(df, feature_importance, threshold):
    # Determine columns to drop based on feature importance
    columns_to_drop = feature_importance[feature_importance['Importance'] < threshold]['Feature'].tolist()
    
    # Drop columns that are not necessary
    df = df.drop(columns=columns_to_drop)
    
    return df
def regression_main(df, importance_threshold, r2_threshold):
    # Preprocess data
    df = preprocess_data(df)
    
    # Separate features and target for regression
    X_reg = df.iloc[:, :-1]  # Features (all columns except the last one)
    y_reg = df.iloc[:, -1]   # Target (last column) for regression
    
    # Apply feature scaling for regression
    scaler_reg = StandardScaler()
    X_scaled_reg = scaler_reg.fit_transform(X_reg)
    
    # Train a Random Forest Regressor to get feature importance
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_scaled_reg, y_reg)
    
    # Calculate feature importance for regression
    feature_importance_reg = pd.DataFrame({'Feature': X_reg.columns, 'Importance': rf_reg.feature_importances_})
    feature_importance_reg = feature_importance_reg.sort_values(by='Importance', ascending=False)
    
    # Drop columns based on importance threshold for regression
    df_reg = drop_by_importance_threshold(df, feature_importance_reg, importance_threshold)
    X_reg=df_reg.iloc[:,:-1]
    y_reg=df_reg.iloc[:,-1]
    scaler1_reg = StandardScaler()
    X_scaled1_reg = scaler1_reg.fit_transform(X_reg)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled_reg, y_reg, test_size=0.2, random_state=42)
    prev_r2=0
    while True:
        # Train and evaluate Random Forest Classifier
        rf_reg.fit(X_train_reg, y_train_reg)
        rf_r2 = evaluate_regressor(rf_reg, X_test_reg, y_test_reg)
        
        # Check if accuracy change is less than the threshold
        if abs(prev_r2-rf_r2) > r2_threshold:
            break
        
        # Drop the column with the least importance
        least_important_feature = feature_importance_reg.iloc[-1]['Feature']
        df_reg = df_reg.drop(columns=[least_important_feature])
        X_reg=df_reg.iloc[:,:-1]
        y_reg=df_reg.iloc[:,-1]
        scaler1_reg = StandardScaler()
        X_scaled1_reg = scaler1_reg.fit_transform(X_reg)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled_reg, y_reg, test_size=0.2, random_state=42)
        
        # Update feature importance and previous accuracy
        rf_reg.fit(X_train_reg, y_train_reg)
        feature_importance_reg= feature_importance_cls[feature_importance_cls['Feature'] != least_important_feature]
        prev_r2 = rf_r2
    
    # Train and evaluate Random Forest Regressor
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train_reg, y_train_reg)
    rf_r2 = evaluate_regressor(rf_regressor, X_test_reg, y_test_reg)

    # Train and evaluate Linear Regression
    lr_regressor = LinearRegression()
    lr_regressor.fit(X_train_reg, y_train_reg)
    lr_r2 = evaluate_regressor(lr_regressor, X_test_reg, y_test_reg)

    # Train and evaluate Support Vector Machine Regressor
    svm_regressor = SVR()
    svm_regressor.fit(X_train_reg, y_train_reg)
    svm_r2 = evaluate_regressor(svm_regressor, X_test_reg, y_test_reg)

    # Train and evaluate Decision Tree Regressor
    dt_regressor = DecisionTreeRegressor()
    dt_regressor.fit(X_train_reg, y_train_reg)
    dt_r2 = evaluate_regressor(dt_regressor, X_test_reg, y_test_reg)

    # return the best model (highest R^2) for regression and additional information
    best_r2 = max([rf_r2, lr_r2, svm_r2, dt_r2])
    if best_r2 == rf_r2:
        return rf_regressor,df_reg
    elif best_r2 == lr_r2:
        return lr_regressor,df_reg
    elif best_r2 == svm_r2:
        return svm_regressor,df_reg
    elif best_r2 == dt_r2:
        return dt_regressor,df_reg


def classification_main(df, importance_threshold, accuracy_threshold):

    # Preprocess data
    df = preprocess_data(df)
    
    # Separate features and target for classification
    X_cls = df.iloc[:, :-1]  # Features (all columns except the last one)
    y_cls = df.iloc[:, -1]   # Target (last column) for classification
    
    # Apply feature scaling for classification
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)
    
    # Train a Random Forest Classifier to get feature importance
    rf_cls = RandomForestClassifier()
    rf_cls.fit(X_scaled_cls, y_cls)
    
    # Calculate feature importance for classification
    feature_importance_cls = pd.DataFrame({'Feature': X_cls.columns, 'Importance': rf_cls.feature_importances_})
    feature_importan9ce_cls = feature_importance_cls.sort_values(by='Importance', ascending=False)
    
    # Drop columns based on importance threshold for classification
    df_cls = drop_by_importance_threshold(df, feature_importance_cls, importance_threshold)
    
    
    X_cls=df_cls.iloc[:,:-1]
    y_cls=df_cls.iloc[:,-1]
    scaler1_cls = StandardScaler()
    X_scaled1_cls = scaler1_cls.fit_transform(X_cls)
     
    # Split data into training and testing sets for classification
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled1_cls, y_cls, test_size=0.2, random_state=42)
    prev_accuracy=0
    while True:
        # Train and evaluate Random Forest Classifier
        rf_cls.fit(X_train_cls, y_train_cls)
        rf_accuracy = evaluate_classifier(rf_cls, X_test_cls, y_test_cls)
        
        # Check if accuracy change is less than the threshold
        if abs(prev_accuracy-rf_accuracy) > accuracy_threshold:
            break
        
        # Drop the column with the least importance
        least_important_feature = feature_importance_cls.iloc[-1]['Feature']
        df_cls = df_cls.drop(columns=[least_important_feature])
        X_cls=df_cls.iloc[:,:-1]
        y_cls=df_cls.iloc[:,-1]
        scaler2_cls = StandardScaler()
        X_scaled2_cls = scaler1_cls.fit_transform(X_cls)
        X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled2_cls, y_cls, test_size=0.2, random_state=42)
        
        # Update feature importance and previous accuracy
        rf_cls.fit(X_train_cls, y_train_cls)
        feature_importance_cls = feature_importance_cls[feature_importance_cls['Feature'] != least_important_feature]
        prev_accuracy = rf_accuracy

    

    # Train and evaluate Random Forest Classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train_cls, y_train_cls)
    rf_accuracy = evaluate_classifier(rf_classifier, X_test_cls, y_test_cls)

    # Train and evaluate Logistic Regression Classifier
    lr_classifier = LogisticRegression()
    lr_classifier.fit(X_train_cls, y_train_cls)
    lr_accuracy = evaluate_classifier(lr_classifier, X_test_cls, y_test_cls)

    # Train and evaluate Support Vector Machine Classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train_cls, y_train_cls)
    svm_accuracy = evaluate_classifier(svm_classifier, X_test_cls, y_test_cls)

    # Train and evaluate Naive Bayes Classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train_cls, y_train_cls)
    nb_accuracy = evaluate_classifier(nb_classifier, X_test_cls, y_test_cls)

    # Train and evaluate Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train_cls, y_train_cls)
    dt_accuracy = evaluate_classifier(dt_classifier, X_test_cls, y_test_cls)
    # Save the best model (highest accuracy) for classification
    best_accuracy = max([rf_accuracy, lr_accuracy, svm_accuracy, nb_accuracy, dt_accuracy])
    if best_accuracy == rf_accuracy:
        return rf_classifier,df_cls
    elif best_accuracy == lr_accuracy:
        return lr_classifier,df_cls
    elif best_accuracy == svm_accuracy:
        return svm_classifier,df_cls
    elif best_accuracy == nb_accuracy:
        return nb_classifier,df_cls
    elif best_accuracy == dt_accuracy:
        return dt_classifier,df_cls


import pandas as pd
import streamlit as st
import joblib
import base64

from classification_module import classification_main
from regression_module import regression_main

def streamlit_main():
    st.title("Teachable Machine V2")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Task selection
        task = st.radio("Select Task:", ("Regression", "Classification"))

        if task == "Classification":
            if st.button("Start Classification"):
                model, processed_df = classification_main(df, 0.01, 0.001)
                st.write("Model trained successfully!")
                st.write("Download the trained model:")
                model_filename = "trained_model.pkl"
                joblib.dump(model, model_filename)
                st.download_button(
                    label="Download Model",
                    data=open("trained_model.pkl", "rb"),
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )
                csv = processed_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download processed data</a>'
                st.markdown(href, unsafe_allow_html=True)

        elif task == "Regression":
            if st.button("Start Regression"):
                model, processed_df = regression_main(df, 0.01, 0.001)
                st.write("Model trained successfully!")
                st.write("Download the trained model:")
                model_filename = "trained_model.pkl"
                joblib.dump(model, model_filename)
                st.download_button(
                    label="Download Model",
                    data=open("trained_model.pkl", "rb"),
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )
                csv = processed_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download processed data</a>'
                st.markdown(href, unsafe_allow_html=True)



if __name__ == "__main__":
    streamlit_main()
