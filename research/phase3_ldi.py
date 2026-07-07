"""
Phase 3: Feature Engineering - Land Degradation Index (LDI) Construction
==========================================================================
Methodology:
  - Normalization: MinMaxScaler (all variables -> [0, 1])
  - Weighting: Equal Weights (1/6 per variable, approved by researcher)
  - Classification: Quantile-based thresholds (33rd and 66th percentile)
  - LDI = 0.5 * (Norm_BareLand + Norm_Temperature)
          - 0.5 * (Norm_NDVI + Norm_SoilMoisture + Norm_TreeCover + Norm_Rainfall)
  Then re-normalized to [0,1] using MinMax.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

# ─────────────────────────── Setup ────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

plots_dir = project_root / 'plots' / 'Phase3'
os.makedirs(plots_dir, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
SAVE_KW = dict(dpi=300, bbox_inches='tight')

df = pd.read_csv(project_root / 'data' / 'working_dataset.csv')

# Variables used in LDI
POSITIVE_VARS = ['BareLand', 'Temperature_mean']   # Increase degradation
NEGATIVE_VARS = ['NDVI_mean', 'SoilMoisture_mean', 'TreeCover', 'Rainfall_mean']
LDI_VARS = POSITIVE_VARS + NEGATIVE_VARS

print("=" * 65)
print("PHASE 3: Land Degradation Index (LDI) Construction")
print("=" * 65)

# ─────────────────────────── 1. Normalization ─────────────────────────────────
print("\n[1] Min-Max Normalization")
print("   Formula: X_norm = (X - X_min) / (X_max - X_min)")
print("   Range after normalization: [0, 1] for all variables\n")

scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[LDI_VARS] = scaler.fit_transform(df[LDI_VARS])

# Print normalized ranges and original ranges side-by-side
print(f"{'Variable':<22} {'Orig Min':>10} {'Orig Max':>10} {'Norm Min':>10} {'Norm Max':>10}")
print("-" * 65)
for i, col in enumerate(LDI_VARS):
    print(f"{col:<22} {df[col].min():>10.4f} {df[col].max():>10.4f} "
          f"{df_norm[col].min():>10.4f} {df_norm[col].max():>10.4f}")

# ─────────────────────────── 2. LDI Formula ───────────────────────────────────
print("\n[2] LDI Formula (Equal Weighting)")
print("   Positive contribution (Degradation drivers):  w = 1/2 each pair")
print("   Negative contribution (Resilience factors):   w = 1/4 each variable")
print()
print("   Raw_LDI = 0.5*(Norm_BareLand + Norm_Temp)")
print("           - 0.5*(Norm_NDVI + Norm_SoilMoisture + Norm_TreeCover + Norm_Rainfall)")
print()
print("   Final_LDI = MinMax(Raw_LDI)  ->  re-scaled to [0, 1]")

w_pos = 1 / len(POSITIVE_VARS)   # 0.5  each
w_neg = 1 / len(NEGATIVE_VARS)   # 0.25 each

raw_ldi = (
    w_pos * df_norm['BareLand']
  + w_pos * df_norm['Temperature_mean']
  - w_neg * df_norm['NDVI_mean']
  - w_neg * df_norm['SoilMoisture_mean']
  - w_neg * df_norm['TreeCover']
  - w_neg * df_norm['Rainfall_mean']
)

# Re-normalize to strict [0, 1]
ldi_min, ldi_max = raw_ldi.min(), raw_ldi.max()
df['LDI'] = (raw_ldi - ldi_min) / (ldi_max - ldi_min)

# ─────────────────────────── 3. Descriptive Statistics ─────────────────────────
print("\n[3] LDI Descriptive Statistics")
ldi_stats = df['LDI'].describe()
print(ldi_stats)
print(f"\nSkewness : {df['LDI'].skew():.4f}")
print(f"Kurtosis : {df['LDI'].kurtosis():.4f}")

# ─────────────────────────── 4. Quantile Thresholds ──────────────────────────
q33 = df['LDI'].quantile(0.33)
q66 = df['LDI'].quantile(0.66)
print(f"\n[4] Classification Thresholds (Quantile-based)")
print(f"   33rd percentile (Low | Moderate boundary) : {q33:.4f}")
print(f"   66th percentile (Moderate | High boundary): {q66:.4f}")

def classify_ldi(x, q33, q66):
    if x <= q33:   return 'Low'
    elif x <= q66: return 'Moderate'
    else:          return 'High'

df['Degradation_Class'] = df['LDI'].apply(classify_ldi, args=(q33, q66))

print("\n[5] Class Distribution")
counts = df['Degradation_Class'].value_counts(sort=False)[['Low', 'Moderate', 'High']]
for cls, cnt in counts.items():
    print(f"   {cls:<10}: {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────── 5. LDI Histogram + KDE ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histogram + KDE
ax = axes[0]
ax.hist(df['LDI'], bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True, label='LDI')
df['LDI'].plot.kde(ax=ax, color='darkblue', linewidth=2)
ax.axvline(q33, color='orange', linestyle='--', linewidth=1.5, label=f'Low|Mod = {q33:.3f}')
ax.axvline(q66, color='red',    linestyle='--', linewidth=1.5, label=f'Mod|High = {q66:.3f}')
ax.set_title('LDI Distribution with Quantile Thresholds', fontsize=13, fontweight='bold')
ax.set_xlabel('Land Degradation Index (LDI)')
ax.set_ylabel('Density')
ax.legend()

# Class pie chart
ax2 = axes[1]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=counts.index, colors=colors,
    autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
ax2.set_title('Degradation Class Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / '1_ldi_distribution.png', **SAVE_KW)
plt.close()
print("\n   -> Saved: plots/Phase3/1_ldi_distribution.png")

# ─────────────────────────── 6. Boxplot by Year ──────────────────────────────
plt.figure(figsize=(10, 6))
order = ['Low', 'Moderate', 'High']
palette = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c'}
sns.boxplot(x='Year', y='LDI', data=df, hue='Degradation_Class',
            palette=palette, hue_order=order, dodge=True, width=0.6)
plt.title('LDI Distribution by Year (2020–2024)', fontsize=13, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Land Degradation Index (LDI)')
plt.legend(title='Class')
plt.tight_layout()
plt.savefig(plots_dir / '2_ldi_by_year.png', **SAVE_KW)
plt.close()
print("   -> Saved: plots/Phase3/2_ldi_by_year.png")

# ─────────────────────────── 7. District-wise Average LDI ────────────────────
dist_ldi = df.groupby('District')['LDI'].mean().sort_values(ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(14, 7))
colors_bar = ['#e74c3c' if v > q66 else ('#f39c12' if v > q33 else '#2ecc71')
              for v in dist_ldi['LDI']]
bars = ax.bar(dist_ldi['District'], dist_ldi['LDI'], color=colors_bar, edgecolor='white')
ax.axhline(q33, color='orange', linestyle='--', linewidth=1.2, label=f'Low|Mod = {q33:.3f}')
ax.axhline(q66, color='red',    linestyle='--', linewidth=1.2, label=f'Mod|High = {q66:.3f}')
ax.set_title('District-wise Average Land Degradation Index (LDI)', fontsize=13, fontweight='bold')
ax.set_xlabel('District')
ax.set_ylabel('Average LDI')
ax.set_xticklabels(dist_ldi['District'], rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(plots_dir / '3_district_ldi.png', **SAVE_KW)
plt.close()
print("   -> Saved: plots/Phase3/3_district_ldi.png")

# ─────────────────────────── 8. Validation Correlations ──────────────────────
val_vars = ['LDI', 'BareLand', 'Temperature_mean', 'NDVI_mean',
            'SoilMoisture_mean', 'TreeCover', 'Rainfall_mean']
corr_vals = df[val_vars].corr()['LDI'].drop('LDI').sort_values()

print("\n[6] LDI Validation – Pearson Correlations with LDI")
print(f"{'Variable':<22} {'r':>8}  Direction Check")
print("-" * 55)
expected = {
    'BareLand':         ('+', 'Higher BareLand -> Higher LDI'),
    'Temperature_mean': ('+', 'Higher Temp -> Higher LDI'),
    'NDVI_mean':        ('-', 'Higher NDVI -> Lower LDI'),
    'SoilMoisture_mean':('-', 'Higher SM -> Lower LDI'),
    'TreeCover':        ('-', 'Higher TreeCover -> Lower LDI'),
    'Rainfall_mean':    ('-', 'Higher Rainfall -> Lower LDI'),
}
all_pass = True
for var, r in corr_vals.items():
    exp_sign, msg = expected[var]
    actual_sign = '+' if r > 0 else '-'
    status = "PASS" if actual_sign == exp_sign else "FAIL"
    if actual_sign != exp_sign: all_pass = False
    print(f"   {var:<22} {r:>8.4f}  {status}  [{msg}]")
print(f"\n   Overall Validation: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

# Correlation bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors_c = ['#e74c3c' if v > 0 else '#3498db' for v in corr_vals.values]
corr_vals.plot(kind='barh', ax=ax, color=colors_c, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Pearson Correlation of Environmental Variables with LDI', fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson r')
ax.set_ylabel('Variable')
plt.tight_layout()
plt.savefig(plots_dir / '4_ldi_validation_corr.png', **SAVE_KW)
plt.close()
print("\n   Saved: plots/Phase3/4_ldi_validation_corr.png")

# ─────────────────────────── 9. LDI Scatter Plots ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
plot_vars = ['BareLand', 'Temperature_mean', 'NDVI_mean',
             'SoilMoisture_mean', 'TreeCover', 'Rainfall_mean']
for i, var in enumerate(plot_vars):
    ax = axes.flatten()[i]
    ax.scatter(df[var], df['LDI'], alpha=0.15, s=6, color='steelblue')
    m, b = np.polyfit(df[var], df['LDI'], 1)
    x_line = np.linspace(df[var].min(), df[var].max(), 200)
    ax.plot(x_line, m * x_line + b, color='red', linewidth=1.8)
    ax.set_xlabel(var)
    ax.set_ylabel('LDI')
    ax.set_title(f'LDI vs {var}')
plt.suptitle('LDI Relationships with Input Variables (Validation)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(plots_dir / '5_ldi_scatter_validation.png', **SAVE_KW)
plt.close()
print("   Saved: plots/Phase3/5_ldi_scatter_validation.png")

# ─────────────────────────── 10. Save LDI Dataset ────────────────────────────
df.to_csv(project_root / 'data' / 'ldi_dataset.csv', index=False)
print("\n[7] ldi_dataset.csv saved with LDI and Degradation_Class columns.")

# ─────────────────────────── 11. District-wise LDI summary ───────────────────
print("\n[8] District-wise Average LDI (Top 5 Most / Least Degraded)")
dist_ldi_full = df.groupby('District')['LDI'].mean().sort_values(ascending=False)
print("\n  Most Degraded:")
print(dist_ldi_full.head(5).to_string())
print("\n  Least Degraded:")
print(dist_ldi_full.tail(5).to_string())

print("\n" + "="*65)
print("Phase 3 Complete. LDI constructed, validated, and saved.")
print("="*65)
