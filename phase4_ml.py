"""
Phase 4: Machine Learning
==========================
Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
Split:  Stratified 80/20
Scaling: StandardScaler for Logistic Regression only
Tuning:  RandomizedSearchCV for tree-based models
"""

import os, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score,
                             ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')
os.makedirs('plots/Phase4', exist_ok=True)
os.makedirs('models', exist_ok=True)
sns.set_theme(style='whitegrid')
SAVE_KW = dict(dpi=300, bbox_inches='tight')

# ─────────────────────────── 1. Load & Prepare Data ──────────────────────────
df = pd.read_csv('ldi_dataset.csv')

# Label encode District (preserves spatial signal without high dimensionality)
le_district = LabelEncoder()
df['District_Enc'] = le_district.fit_transform(df['District'])

# Label encode target
le_target = LabelEncoder()
CLASS_ORDER = ['Low', 'Moderate', 'High']   # 0, 1, 2
df['Target'] = le_target.fit_transform(df['Degradation_Class'])

# Feature matrix — exclude Grid_ID (identifier), LDI (derived target), 
#                  Degradation_Class (target string), District (replaced by encoded)
EXCLUDE = ['Grid_ID', 'District', 'LDI', 'Degradation_Class', 'Target']
FEATURES = [c for c in df.columns if c not in EXCLUDE]

X = df[FEATURES].copy()
y = df['Target'].values

print("=" * 65)
print("PHASE 4: Machine Learning")
print("=" * 65)
print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")
print(f"Target classes: {le_target.classes_}  (encoded 0, 1, 2)")
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ─────────────────────────── 2. Stratified Train-Test Split ──────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
print("Class balance in test set:")
for cls, name in zip([0,1,2], CLASS_ORDER):
    print(f"  {name}: {(y_test==cls).sum()} ({(y_test==cls).mean()*100:.1f}%)")

# ─────────────────────────── 3. Feature Scaling (LR only) ────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, 'models/standard_scaler.pkl')

# ─────────────────────────── 4. Cross-validation helper ──────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def cv_score(model, X, y):
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    return scores.mean(), scores.std()

# ─────────────────────────── 5a. Logistic Regression ─────────────────────────
print("\n[1/5] Logistic Regression ...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)
y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)
cv_lr      = cv_score(lr, X_train_sc, y_train)
print(f"   CV F1-macro: {cv_lr[0]:.4f} +/- {cv_lr[1]:.4f}")

# ─────────────────────────── 5b. Decision Tree ───────────────────────────────
print("\n[2/5] Decision Tree (RandomizedSearchCV) ...")
dt_param = {'max_depth': [3,5,7,10,15,None],
            'min_samples_split': [2,5,10,20],
            'min_samples_leaf':  [1,2,4,8],
            'criterion': ['gini','entropy']}
dt_base = DecisionTreeClassifier(random_state=42)
dt_rs   = RandomizedSearchCV(dt_base, dt_param, n_iter=30, cv=cv,
                             scoring='f1_macro', n_jobs=-1, random_state=42)
dt_rs.fit(X_train, y_train)
dt = dt_rs.best_estimator_
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)
cv_dt     = cv_score(dt, X_train, y_train)
print(f"   Best params: {dt_rs.best_params_}")
print(f"   CV F1-macro: {cv_dt[0]:.4f} +/- {cv_dt[1]:.4f}")

# ─────────────────────────── 5c. Random Forest ───────────────────────────────
print("\n[3/5] Random Forest (RandomizedSearchCV) ...")
rf_param = {'n_estimators': [100,200,300],
            'max_depth': [5,10,15,None],
            'min_samples_split': [2,5,10],
            'min_samples_leaf':  [1,2,4],
            'max_features': ['sqrt','log2']}
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_rs   = RandomizedSearchCV(rf_base, rf_param, n_iter=20, cv=cv,
                             scoring='f1_macro', n_jobs=-1, random_state=42)
rf_rs.fit(X_train, y_train)
rf = rf_rs.best_estimator_
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)
cv_rf     = cv_score(rf, X_train, y_train)
print(f"   Best params: {rf_rs.best_params_}")
print(f"   CV F1-macro: {cv_rf[0]:.4f} +/- {cv_rf[1]:.4f}")

# ─────────────────────────── 5d. Gradient Boosting ───────────────────────────
print("\n[4/5] Gradient Boosting (RandomizedSearchCV) ...")
gb_param = {'n_estimators': [100,200,300],
            'max_depth': [3,4,5],
            'learning_rate': [0.05,0.1,0.2],
            'subsample': [0.7,0.8,1.0],
            'min_samples_leaf': [1,2,4]}
gb_base = GradientBoostingClassifier(random_state=42)
gb_rs   = RandomizedSearchCV(gb_base, gb_param, n_iter=20, cv=cv,
                             scoring='f1_macro', n_jobs=-1, random_state=42)
gb_rs.fit(X_train, y_train)
gb = gb_rs.best_estimator_
y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)
cv_gb     = cv_score(gb, X_train, y_train)
print(f"   Best params: {gb_rs.best_params_}")
print(f"   CV F1-macro: {cv_gb[0]:.4f} +/- {cv_gb[1]:.4f}")

# ─────────────────────────── 5e. XGBoost ─────────────────────────────────────
try:
    from xgboost import XGBClassifier
    print("\n[5/5] XGBoost (RandomizedSearchCV) ...")
    xgb_param = {'n_estimators': [100,200,300],
                 'max_depth': [3,4,5,6],
                 'learning_rate': [0.05,0.1,0.2],
                 'subsample': [0.7,0.8,1.0],
                 'colsample_bytree': [0.7,0.8,1.0],
                 'reg_alpha': [0,0.1,0.5],
                 'reg_lambda': [1,1.5,2]}
    xgb_base = XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
    xgb_rs   = RandomizedSearchCV(xgb_base, xgb_param, n_iter=20, cv=cv,
                                  scoring='f1_macro', n_jobs=-1, random_state=42)
    xgb_rs.fit(X_train, y_train)
    xgb = xgb_rs.best_estimator_
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)
    cv_xgb     = cv_score(xgb, X_train, y_train)
    print(f"   Best params: {xgb_rs.best_params_}")
    print(f"   CV F1-macro: {cv_xgb[0]:.4f} +/- {cv_xgb[1]:.4f}")
    XGB_AVAILABLE = True
except ImportError:
    print("\n[5/5] XGBoost not available — skipping.")
    XGB_AVAILABLE = False

# ─────────────────────────── 6. Evaluation Helper ────────────────────────────
def evaluate(name, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    rpt  = classification_report(y_true, y_pred, target_names=CLASS_ORDER, output_dict=True)
    f1   = rpt['macro avg']['f1-score']
    prec = rpt['macro avg']['precision']
    rec  = rpt['macro avg']['recall']
    auc  = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    return {'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1-Macro': f1, 'ROC-AUC': auc}

results = []
results.append(evaluate('Logistic Regression', y_test, y_pred_lr, y_prob_lr))
results.append(evaluate('Decision Tree',       y_test, y_pred_dt, y_prob_dt))
results.append(evaluate('Random Forest',       y_test, y_pred_rf, y_prob_rf))
results.append(evaluate('Gradient Boosting',   y_test, y_pred_gb, y_prob_gb))
if XGB_AVAILABLE:
    results.append(evaluate('XGBoost',         y_test, y_pred_xgb, y_prob_xgb))

results_df = pd.DataFrame(results).sort_values('F1-Macro', ascending=False)
results_df.to_csv('models/model_comparison.csv', index=False)

print("\n" + "=" * 65)
print("MODEL COMPARISON TABLE")
print("=" * 65)
print(results_df.to_string(index=False, float_format='{:.4f}'.format))

# ─────────────────────────── 7. Confusion Matrices ───────────────────────────
all_models = [('Logistic Regression', y_pred_lr),
              ('Decision Tree',       y_pred_dt),
              ('Random Forest',       y_pred_rf),
              ('Gradient Boosting',   y_pred_gb)]
if XGB_AVAILABLE:
    all_models.append(('XGBoost', y_pred_xgb))

n_models = len(all_models)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
for ax, (name, y_pred) in zip(axes, all_models):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=10, fontweight='bold')
plt.suptitle('Confusion Matrices (All Models)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/Phase4/1_confusion_matrices.png', **SAVE_KW)
plt.close()

# ─────────────────────────── 8. Model Comparison Bar Chart ───────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Macro', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15
fig, ax = plt.subplots(figsize=(14, 7))
colors = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6']
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=colors[i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=15)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/Phase4/2_model_comparison.png', **SAVE_KW)
plt.close()

# ─────────────────────────── 9. Error Analysis ───────────────────────────────
# Best model
best_name  = results_df.iloc[0]['Model']
model_map  = {'Logistic Regression': y_pred_lr, 'Decision Tree': y_pred_dt,
              'Random Forest': y_pred_rf, 'Gradient Boosting': y_pred_gb}
if XGB_AVAILABLE: model_map['XGBoost'] = y_pred_xgb
best_pred  = model_map[best_name]

# Classification report for best model
print(f"\n--- Detailed Report: {best_name} ---")
print(classification_report(y_test, best_pred, target_names=CLASS_ORDER))

# Add predictions to test set for error analysis
df_test = X_test.copy()
df_test['y_true'] = y_test
df_test['y_pred'] = best_pred
df_test['correct'] = (y_test == best_pred).astype(int)
df_test['District'] = le_district.inverse_transform(df_test['District_Enc'].astype(int))

# District-wise error rate
district_errors = df_test.groupby('District').apply(
    lambda g: 1 - g['correct'].mean()
).sort_values(ascending=False).reset_index()
district_errors.columns = ['District', 'Error_Rate']

print("\n--- District-wise Error Rate (Top 10) ---")
print(district_errors.head(10).to_string(index=False))

plt.figure(figsize=(14, 6))
colors_err = ['#e74c3c' if v > 0.15 else '#f39c12' if v > 0.10 else '#2ecc71'
              for v in district_errors['Error_Rate']]
plt.bar(district_errors['District'], district_errors['Error_Rate'], color=colors_err, edgecolor='white')
plt.title(f'District-wise Classification Error Rate ({best_name})', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Error Rate')
plt.axhline(district_errors['Error_Rate'].mean(), color='black', linestyle='--',
            linewidth=1.2, label='Mean Error Rate')
plt.legend()
plt.tight_layout()
plt.savefig('plots/Phase4/3_district_error.png', **SAVE_KW)
plt.close()

# ─────────────────────────── 10. Save Best Model ─────────────────────────────
best_model_obj = None
if   best_name == 'Logistic Regression':  best_model_obj = lr
elif best_name == 'Decision Tree':        best_model_obj = dt
elif best_name == 'Random Forest':        best_model_obj = rf
elif best_name == 'Gradient Boosting':    best_model_obj = gb
elif best_name == 'XGBoost':             best_model_obj = xgb

joblib.dump(best_model_obj, f'models/best_model_{best_name.replace(" ","_")}.pkl')
joblib.dump(le_district,    'models/le_district.pkl')
joblib.dump(le_target,      'models/le_target.pkl')

# Save predictions
df_preds = pd.DataFrame({'y_true_label': le_target.inverse_transform(y_test),
                          'y_pred_label': le_target.inverse_transform(best_pred)})
df_preds['correct'] = (df_preds['y_true_label'] == df_preds['y_pred_label'])
df_preds.to_csv('models/best_model_predictions.csv', index=False)

print(f"\nBest model: {best_name}")
print("Saved: models/best_model_*.pkl, models/best_model_predictions.csv")
print("Saved: plots/Phase4/*.png")
print("\n" + "="*65)
print("Phase 4 Complete.")
print("="*65)
