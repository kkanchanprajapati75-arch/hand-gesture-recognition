import os
import csv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

GESTURES = [
    "Open_Palm", "Fist", "Thumbs_Up", "Thumbs_Down",
    "Victory", "Three_Fingers", "Four_Fingers",
    "OK_Sign", "Rock_Sign", "Call_Me"
]

X, y = [], []

for gesture in GESTURES:
    path = f"dataset/{gesture}/landmarks.csv"
    if not os.path.exists(path):
        print("Missing:", gesture)
        continue

    with open(path) as f:
        for row in csv.reader(f):
            if len(row) == 63:
                X.append([float(v) for v in row])
                y.append(gesture)

print("Total samples:", len(X))
print("Gestures:", sorted(set(y)))

if len(set(y)) != 10:
    raise SystemExit("Error: All 10 gestures need training data!")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

joblib.dump(model, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
