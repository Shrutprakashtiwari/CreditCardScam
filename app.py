import pandas as pd
df=pd.read_csv("creditcard.csv")
df.head()
from matplotlib import pyplot as plt
x=df.drop(columns=["Class"])
y=df["Class"]
df = df.dropna()
# plt.plot(x,y)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_test = pd.DataFrame(x_test).dropna()

model=LogisticRegression(class_weight='balanced')
model.fit(x_train,y_train)
y_proba=model.predict_proba(x_test)[:,1]
y_pred=(y_proba>0.85).astype(int)
# y_pred=model.predict(x_test)
print(y_pred)
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
model1=RandomForestClassifier(n_estimators=100, class_weight='balanced')
model1.fit(x_train,y_train)
y_proba=model1.predict_proba(x_test)[:,1]
y_pred=(y_proba>0.2).astype(int)
print(y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')

scores = cross_val_score(rf, x, y, cv=5, scoring='recall')
print(scores)
print("Mean Recall:", scores.mean())
from xgboost import XGBClassifier
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=3,
    eval_metric='logloss',
    random_state=42,
    class_weight='balanced')
model.fit(x_train, y_train)
y_proba = model.predict_proba(x_test)[:,1]
y_pred = (y_proba > 0.2).astype(int)
print(y_pred)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test, y_pred))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
PrecisionRecallDisplay.from_predictions(y_test,y_proba)

import numpy as np

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("Best Threshold:", best_threshold)
print("Precision:", precision[best_idx])
print("Recall:", recall[best_idx])
idx=np.where(recall>0.9)
print(thresholds[idx])

