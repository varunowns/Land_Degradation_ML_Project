import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

from pathlib import Path

def run_phase2_eda():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    df = pd.read_csv(project_root / 'data' / 'working_dataset.csv')
    
    plots_dir = project_root / 'plots' / 'EDA'
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    env_cols = ['NDVI_mean', 'Rainfall_mean', 'Temperature_mean', 'SoilMoisture_mean']
    lulc_cols = ['BareLand', 'Builtup', 'Cropland', 'Grassland', 'TreeCover', 'Water', 'Wetland', 'Shrubland']
    num_cols = env_cols + lulc_cols + ['Area_km2']
    
    with open(project_root / 'results' / 'eda_stats.txt', 'w') as f:
        f.write("=== 1. Dataset Overview ===\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Data Types:\n{df.dtypes}\n")
        f.write(f"Summary Stats:\n{df[num_cols].describe().T}\n\n")
        
        f.write("=== 2. Univariate Analysis (Distributions) ===\n")
        for col in num_cols:
            f.write(f"{col} - Mean: {df[col].mean():.3f}, Median: {df[col].median():.3f}, Skewness: {skew(df[col]):.3f}, Kurtosis: {kurtosis(df[col]):.3f}\n")
        
        # Plot Hist/KDE
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, col in enumerate(env_cols):
            sns.histplot(df[col], kde=True, ax=axes.flatten()[i], color='teal', bins=50)
            axes.flatten()[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(plots_dir / '1_env_distributions.png', dpi=300)
        plt.close()
        
        # Plot Boxplots for Outliers
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for i, col in enumerate(env_cols):
            sns.boxplot(y=df[col], ax=axes.flatten()[i], color='salmon')
            axes.flatten()[i].set_title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(plots_dir / '2_env_boxplots.png', dpi=300)
        plt.close()
        
        # LULC distributions
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i, col in enumerate(lulc_cols):
            sns.histplot(df[col], kde=True, ax=axes.flatten()[i], bins=30, color='purple')
            axes.flatten()[i].set_title(col)
        plt.tight_layout()
        plt.savefig(plots_dir / '3_lulc_distributions.png', dpi=300)
        plt.close()
        
        f.write("\n=== 3. Correlation Analysis ===\n")
        pearson_corr = df[num_cols].corr(method='pearson')
        spearman_corr = df[num_cols].corr(method='spearman')
        
        # Plot Pearson Heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, vmin=-1, vmax=1)
        plt.title('Pearson Correlation Matrix')
        plt.tight_layout()
        plt.savefig(plots_dir / '4_pearson_heatmap.png', dpi=300)
        plt.close()
        
        # Plot Spearman Heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.2f', cmap='viridis', square=True, vmin=-1, vmax=1)
        plt.title('Spearman Correlation Matrix')
        plt.tight_layout()
        plt.savefig(plots_dir / '5_spearman_heatmap.png', dpi=300)
        plt.close()
        
        # Highlight highly correlated |r| > 0.7
        f.write("Highly Correlated Pairs (|r| > 0.7, Pearson):\n")
        for i in range(len(pearson_corr.columns)):
            for j in range(i):
                if abs(pearson_corr.iloc[i, j]) > 0.7:
                    f.write(f"  {pearson_corr.columns[i]} & {pearson_corr.columns[j]}: {pearson_corr.iloc[i, j]:.3f}\n")
        
        f.write("\n=== 4. Temporal Analysis ===\n")
        # Line plots for trends
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        sns.lineplot(x='Year', y='NDVI_mean', data=df, ax=axes[0], marker='o', ci='sd', color='green')
        axes[0].set_title('Year-wise NDVI Trend')
        sns.lineplot(x='Year', y='Rainfall_mean', data=df, ax=axes[1], marker='o', ci='sd', color='blue')
        axes[1].set_title('Year-wise Rainfall Trend')
        sns.lineplot(x='Year', y='Temperature_mean', data=df, ax=axes[2], marker='o', ci='sd', color='red')
        axes[2].set_title('Year-wise Temperature Trend')
        plt.tight_layout()
        plt.savefig(plots_dir / '6_temporal_trends.png', dpi=300)
        plt.close()
        
        f.write("\n=== 5. Spatial Analysis (District Extremes) ===\n")
        dist_stats = df.groupby('District')[['NDVI_mean', 'Rainfall_mean', 'Temperature_mean', 'BareLand']].mean()
        f.write(f"Highest NDVI: {dist_stats['NDVI_mean'].idxmax()} ({dist_stats['NDVI_mean'].max():.3f})\n")
        f.write(f"Lowest NDVI: {dist_stats['NDVI_mean'].idxmin()} ({dist_stats['NDVI_mean'].min():.3f})\n")
        f.write(f"Highest Rainfall: {dist_stats['Rainfall_mean'].idxmax()} ({dist_stats['Rainfall_mean'].max():.3f})\n")
        f.write(f"Lowest Rainfall: {dist_stats['Rainfall_mean'].idxmin()} ({dist_stats['Rainfall_mean'].min():.3f})\n")
        f.write(f"Highest Temp: {dist_stats['Temperature_mean'].idxmax()} ({dist_stats['Temperature_mean'].max():.3f})\n")
        f.write(f"Highest BareLand: {dist_stats['BareLand'].idxmax()} ({dist_stats['BareLand'].max():.3f})\n")
        
        # District Boxplots
        plt.figure(figsize=(18, 8))
        sns.boxplot(x='District', y='NDVI_mean', data=df, palette='Set3')
        plt.title('District-wise NDVI Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / '7_district_ndvi.png', dpi=300)
        plt.close()
        
        f.write("\n=== 6. Outlier Analysis (IQR Method) ===\n")
        for col in env_cols + ['BareLand', 'TreeCover']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            f.write(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)\n")
            
        # 7. Relationship Analysis (Scatter plots)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        vars_to_plot = ['Rainfall_mean', 'Temperature_mean', 'SoilMoisture_mean', 'TreeCover', 'BareLand', 'Cropland']
        for i, var in enumerate(vars_to_plot):
            sns.scatterplot(x=var, y='NDVI_mean', data=df, ax=axes.flatten()[i], alpha=0.3, s=10)
            sns.regplot(x=var, y='NDVI_mean', data=df, ax=axes.flatten()[i], scatter=False, color='red')
            axes.flatten()[i].set_title(f'NDVI vs {var}')
        plt.tight_layout()
        plt.savefig(plots_dir / '8_relationships.png', dpi=300)
        plt.close()
        
        # 8. Multivariate Analysis (Pairplot)
        pair_vars = ['NDVI_mean', 'Rainfall_mean', 'Temperature_mean', 'BareLand']
        pairplot = sns.pairplot(df[pair_vars], corner=True, diag_kind='kde', plot_kws={'alpha': 0.2, 's': 5})
        pairplot.fig.suptitle('Pairplot of Core Variables', y=1.02)
        plt.savefig(plots_dir / '9_pairplot.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    run_phase2_eda()
