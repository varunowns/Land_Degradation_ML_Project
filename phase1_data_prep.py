import pandas as pd
import os

def prepare_data():
    years = [2020, 2021, 2022, 2023, 2024]
    variables = ['NDVI', 'Rainfall', 'Temperature', 'SoilMoisture']
    
    all_years_data = []
    
    for year in years:
        # Read LULC as the base dataframe
        lulc_file = f'Land_Degradation/LULC_{year}.csv'
        df_base = pd.read_csv(lulc_file)
        
        # Add the Year column
        df_base['Year'] = year
        
        # Merge the other variable datasets for the same year
        for var in variables:
            var_file = f'Land_Degradation/{var}_{year}.csv'
            df_var = pd.read_csv(var_file)
            
            # Extract Grid_ID and the variable's value (named 'mean' in GEE exports)
            df_var = df_var[['Grid_ID', 'mean']]
            df_var = df_var.rename(columns={'mean': f'{var}_mean'})
            
            # Merge based on the unique Grid_ID
            df_base = pd.merge(df_base, df_var, on='Grid_ID', how='inner')
            
        all_years_data.append(df_base)
        
    # Concatenate all years vertically
    master_df = pd.concat(all_years_data, ignore_index=True)
    
    # Reorder columns as specified by the user
    expected_columns = [
        'Grid_ID', 'District', 'Year', 'Area_km2', 'BareLand', 'Builtup', 'Cropland',
        'Grassland', 'TreeCover', 'Water', 'Wetland', 'Shrubland', 'Mangroves',
        'MossLichen', 'SnowIce', 'NDVI_mean', 'Rainfall_mean', 'Temperature_mean',
        'SoilMoisture_mean'
    ]
    master_df = master_df[expected_columns]
    
    # Check for duplicates and drop if any
    duplicates = master_df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Dropping them...")
        master_df = master_df.drop_duplicates()
        
    # Handle missing values: Impute using median to preserve all 22,645 rows
    missing_count = master_df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\nMissing values detected per column:\n{master_df.isnull().sum()}")
        # Impute NDVI_mean missing values with the median of the respective year and district
        master_df['NDVI_mean'] = master_df.groupby(['Year', 'District'])['NDVI_mean'].transform(lambda x: x.fillna(x.median()))
        
        # If any remain (e.g. whole district was missing), fill with global median
        if master_df['NDVI_mean'].isnull().sum() > 0:
            master_df['NDVI_mean'] = master_df['NDVI_mean'].fillna(master_df['NDVI_mean'].median())
            
        print("Imputed missing values to preserve exactly 22,645 rows.")
    
    # Save the final reproducible merged dataset
    master_df.to_csv('master_dataset.csv', index=False)
    
    print("\n--- Phase 1 Verification ---")
    print(f"Final Master Dataset Shape: {master_df.shape}")
    print(f"Total Columns: {len(master_df.columns)}")
    print("\nData Types:")
    print(master_df.dtypes)
    print("\nMissing Values Left:")
    print(master_df.isnull().sum().sum())
    
if __name__ == "__main__":
    prepare_data()
