import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTEENN

def clean_and_split(file_path):
    data = pd.read_csv(file_path)
    
    # 1. Handle Zeros as Missing
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
    
    # 2. Log Transform Skewed Features (Reduces the impact of outliers)
    # Insulin and Pedigree are usually highly skewed.
    data['Insulin'] = np.log1p(data['Insulin'])
    data['DiabetesPedigreeFunction'] = np.log1p(data['DiabetesPedigreeFunction'])

    # 3. Feature Engineering: High Risk Indicator
    # Giving the model a "hint" based on medical knowledge
    data['Is_Obese_High_Glucose'] = ((data['BMI'] > 30) & (data['Glucose'] > 140)).astype(int)

    # 4. Separate Features and Target
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # 5. The Clean Split (NO LEAKAGE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 6. Imputation (Fit on Train only)
    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 7. Scaling (Fit on Train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 8. SMOTE ONLY ON TRAIN
    smoteenn = SMOTEENN(random_state=42)
    X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)

    # 9. Reshape for BiLSTM
    X_train_final = X_train_res.reshape((X_train_res.shape[0], 1, X_train_res.shape[1]))
    X_test_final = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train_final, X_test_final, y_train_res, y_test, scaler