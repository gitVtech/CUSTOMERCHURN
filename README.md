# CUSTOMERCHURN
sourcecode
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("synthetic_churn.csv")

y = df["Churn"]
X = df.drop(columns=["CustomerID", "Churn"])

le = LabelEncoder()
y = le.fit_transform(y)

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

print("✅ Data prepared. Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smoteenn = SMOTEENN()
X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

print("✅ After SMOTEENN: ", X_resampled.shape, " | Class balance:", pd.Series(y_resampled).value_counts().to_dict())

dt_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=6, min_samples_leaf=8)
dt_model.fit(X_resampled, y_resampled)
dt_pred = dt_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, criterion="gini", random_state=42, max_depth=6, min_samples_leaf=8)
rf_model.fit(X_resampled, y_resampled)
rf_pred = rf_model.predict(X_test)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_resampled, y_resampled)
svm_pred = svm_model.predict(X_test)

models = {
    "Decision Tree": (dt_model, dt_pred),
    "Random Forest": (rf_model, rf_pred),
    "SVM": (svm_model, svm_pred)
}

results = []

for name, (model, y_pred) in models.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1]) if hasattr(model, "predict_proba") else None

    print(f"\n📊 {name} Results:")
    print(f"   ✅ Accuracy: {acc*100:.2f}%")
    print(f"   ✅ Precision: {prec*100:.2f}%")
    print(f"   ✅ Recall: {rec*100:.2f}%")
    print(f"   ✅ F1-Score: {f1*100:.2f}%")
    if roc is not None:
        print(f"   ✅ ROC-AUC: {roc:.3f}")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc if roc is not None else 0
    })

metrics_df = pd.DataFrame(results)

plt.figure(figsize=(10,6))

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for metric, color in zip(metrics, colors):
    plt.plot(metrics_df["Model"], metrics_df[metric],
             marker="o", markersize=8, linewidth=2,
             color=color, label=metric)

    for x, y in zip(metrics_df["Model"], metrics_df[metric]):
        plt.text(x, y+0.02, f"{y:.2f}", ha="center", fontsize=9, color=color)

plt.title("Model Comparison Across All Metrics", fontsize=16, fontweight="bold")
plt.xlabel("Models", fontsize=13, fontweight="bold")
plt.ylabel("Score", fontsize=13, fontweight="bold")
plt.ylim(0, 1.1)
plt.legend(title="Metrics", fontsize=10, title_fontsize=11, loc="lower right", frameon=True, shadow=True)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
