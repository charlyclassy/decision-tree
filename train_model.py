"""
UK Weather Decision Tree Model
================================
Trains a Decision Tree classifier on UK weather data (2023)
to predict weather conditions from temperature, humidity, and precipitation.

Dataset: 3,285 daily records across 9 UK cities
Features: temperature_celsius, humidity_percent, precipitation_mm
Target:   weather_condition (8 classes)
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv('uk_weather_data.csv')

print("=" * 60)
print("UK WEATHER DECISION TREE MODEL")
print("=" * 60)
print(f"\nDataset shape: {df.shape}")
print(f"Cities: {sorted(df['city'].unique().tolist())}")
print(f"\nWeather condition distribution:")
print(df['weather_condition'].value_counts())

# ── 2. Prepare Features & Target ─────────────────────────────────────────────
FEATURES = ['temperature_celsius', 'humidity_percent', 'precipitation_mm']
TARGET = 'weather_condition'

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeatures: {FEATURES}")
print(f"Target classes: {sorted(y.unique().tolist())}")

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── 4. Train Decision Tree ────────────────────────────────────────────────────
clf = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)
clf.fit(X_train, y_train)

print(f"\nTree depth:  {clf.get_depth()}")
print(f"Tree leaves: {clf.get_n_leaves()}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 6. Feature Importance ─────────────────────────────────────────────────────
importance = dict(zip(FEATURES, clf.feature_importances_))
print("\nFeature Importances:")
for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f} ({imp*100:.1f}%)")

# ── 7. Print Tree Rules ───────────────────────────────────────────────────────
print("\nDecision Tree Rules:")
print(export_text(clf, feature_names=FEATURES))

# ── 8. Visualise Tree ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 10))
plot_tree(
    clf,
    feature_names=FEATURES,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    fontsize=7,
    ax=ax
)
plt.title("UK Weather Decision Tree (max_depth=6)", fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
print("\nTree diagram saved to: decision_tree.png")

# ── 9. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=clf.classes_, yticklabels=clf.classes_, ax=ax
)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved to: confusion_matrix.png")

# ── 10. Export Tree as JSON (for web deployment) ──────────────────────────────
from sklearn.tree import _tree

def tree_to_dict(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    def recurse(node):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            return {
                "feature": feature_name[node],
                "threshold": round(float(tree_.threshold[node]), 2),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            values = tree_.value[node][0]
            total = sum(values)
            pred_class = class_names[list(values).index(max(values))]
            confidence = round(max(values) / total * 100, 1)
            probs = {
                class_names[i]: round(float(v) / total, 3)
                for i, v in enumerate(values) if v > 0
            }
            return {
                "prediction": pred_class,
                "confidence": confidence,
                "probabilities": probs
            }

    return recurse(0)


tree_dict = tree_to_dict(clf, FEATURES, clf.classes_.tolist())

with open('tree_model.json', 'w') as f:
    json.dump({
        "tree": tree_dict,
        "classes": clf.classes_.tolist(),
        "features": FEATURES,
        "accuracy": round(accuracy, 3),
        "feature_importances": importance
    }, f, indent=2)

print("Model exported to: tree_model.json")

# ── 11. Predict a sample ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
samples = [
    {"temperature_celsius": -2.0, "humidity_percent": 85.0, "precipitation_mm": 8.0},
    {"temperature_celsius": 18.0, "humidity_percent": 65.0, "precipitation_mm": 0.0},
    {"temperature_celsius": 10.0, "humidity_percent": 92.0, "precipitation_mm": 0.5},
    {"temperature_celsius": 5.0,  "humidity_percent": 78.0, "precipitation_mm": 12.0},
]
for s in samples:
    pred = clf.predict(pd.DataFrame([s]))[0]
    proba = clf.predict_proba(pd.DataFrame([s]))[0]
    conf = max(proba) * 100
    print(f"  Temp={s['temperature_celsius']}°C  Humidity={s['humidity_percent']}%  "
          f"Precip={s['precipitation_mm']}mm  →  {pred} ({conf:.1f}%)")
