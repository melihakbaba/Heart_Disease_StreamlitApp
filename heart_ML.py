# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:18:18 2024

@author: melih
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

heart_data = pd.read_csv("heart.csv")

heart_data.head()

heart_data.info()


X = heart_data.drop(columns=['output'])  # Features

y = heart_data['output']  # Target  

# 1: High Chance - 0: Low Chance

X["thall"].replace(0,1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


joblib.dump(model, 'heart_disease_model.pkl')
