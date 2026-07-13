import pandas as pd

df = pd.read_csv('master_dataset.csv')

print('--- 1. Complete List of Column Names ---')
print(df.columns.tolist())

print('\n--- 2a. df.info() ---')
df.info()

print('\n--- 2b. df.describe(include="all") ---')
print(df.describe(include='all').to_string())

print('\n--- 3. Report ---')
print(f'Number of rows: {df.shape[0]}')
print(f'Number of columns: {df.shape[1]}')
print(f'Missing values per column:\n{df.isnull().sum()}')
print(f'Duplicate rows: {df.duplicated().sum()}')
print(f'Data types:\n{df.dtypes}')

print('\n--- 4. Verification Check ---')
print(f'Total observations exactly 22645? {df.shape[0] == 22645}')
print(f'All 5 years present? {sorted(df["Year"].unique().tolist()) == [2020, 2021, 2022, 2023, 2024]}')
print(f'Number of districts: {df["District"].nunique()}')
print(f'Any duplicates based on Grid_ID and Year? {df.duplicated(subset=["Grid_ID", "Year"]).sum()}')

print('\n--- 5. Impossible Values Check ---')
print(f'Negative Rainfall? {(df["Rainfall_mean"] < 0).sum()} rows')
print(f'NDVI outside [-1, 1]? {((df["NDVI_mean"] < -1) | (df["NDVI_mean"] > 1)).sum()} rows')
print(f'Soil Moisture outside [0, 1] (or typical bounds)? {((df["SoilMoisture_mean"] < 0) | (df["SoilMoisture_mean"] > 1)).sum()} rows')

lulc_cols = ['BareLand', 'Builtup', 'Cropland', 'Grassland', 'TreeCover', 'Water', 'Wetland', 'Shrubland', 'Mangroves', 'MossLichen', 'SnowIce']
df['LULC_sum'] = df[lulc_cols].sum(axis=1)
print(f'LULC sum outside [0, 105%]? {((df["LULC_sum"] < 0) | (df["LULC_sum"] > 105)).sum()} rows')
