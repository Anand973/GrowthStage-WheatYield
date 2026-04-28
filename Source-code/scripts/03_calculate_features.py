"""
SCRIPT 3: Calculate Growth Stage Features
Run this third!

Purpose: Calculate climate features for each growth stage
This is THE CORE INNOVATION of your research!
Input: climate_clean.csv
Output: climate_features.csv
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# IMPORTANT: CHANGE THIS PATH TO YOUR ACTUAL FOLDER LOCATION!
# ============================================================================

PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'  # ← CHANGE THIS!

# ============================================================================

os.chdir(PROJECT_DIR)

print("="*70)
print("SCRIPT 3: CALCULATING GROWTH STAGE FEATURES")
print("="*70)
print(f"\nWorking directory: {os.getcwd()}")

# Load climate data
climate = pd.read_csv('data/processed/climate_clean.csv')
climate['Date'] = pd.to_datetime(climate['Date'])

print(f"\n✅ Loaded climate data: {len(climate):,} rows")

# Growth stage definitions
GROWTH_STAGES = {
    'flowering': {
        'start_month': 2, 'start_day': 1,
        'end_month': 2, 'end_day': 15,
        'description': 'Flowering (Feb 1-15) - most heat sensitive'
    },
    'grain_filling': {
        'start_month': 2, 'start_day': 16,
        'end_month': 3, 'end_day': 15,
        'description': 'Grain filling (Feb 16 - Mar 15) - moisture critical'
    },
    'maturation': {
        'start_month': 3, 'start_day': 16,
        'end_month': 4, 'end_day': 10,
        'description': 'Maturation (Mar 16 - Apr 10) - grain hardening'
    }
}

print("\n📅 Growth Stage Definitions:")
for stage, info in GROWTH_STAGES.items():
    print(f"   {stage.title()}: {info['description']}")

def get_stage_data(df, year, stage_info):
    """Extract climate data for a specific growth stage"""
    start_month = stage_info['start_month']
    start_day = stage_info['start_day']
    end_month = stage_info['end_month']
    end_day = stage_info['end_day']
    
    mask = (
        (df['Year'] == year) &
        (
            ((df['Month'] == start_month) & (df['Day'] >= start_day)) |
            ((df['Month'] > start_month) & (df['Month'] < end_month)) |
            ((df['Month'] == end_month) & (df['Day'] <= end_day))
        )
    )
    return df[mask]

def calculate_stage_features(stage_data, stage_name):
    """Calculate climate features for a growth stage"""
    if len(stage_data) == 0:
        return {f'{stage_name}_tmax_mean': np.nan}
    
    features = {
        f'{stage_name}_tmax_mean': stage_data['Tmax'].mean(),
        f'{stage_name}_tmin_mean': stage_data['Tmin'].mean(),
        f'{stage_name}_tmax_max': stage_data['Tmax'].max(),
        f'{stage_name}_tmax_std': stage_data['Tmax'].std(),
        f'{stage_name}_day_night_diff': (stage_data['Tmax'] - stage_data['Tmin']).mean(),
        f'{stage_name}_rainfall_total': stage_data['Rainfall'].sum(),
        f'{stage_name}_rainy_days': (stage_data['Rainfall'] > 0).sum(),
        f'{stage_name}_extreme_heat_days': (stage_data['Tmax'] > 35).sum()
    }
    
    if stage_name in ['flowering', 'grain_filling']:
        features[f'{stage_name}_heat_stress_days'] = (stage_data['Tmax'] > 30).sum()
    
    return features

# Calculate features for each district-year
all_features = []

districts = climate['District'].unique()
years = sorted(climate['Year'].unique())

print(f"\n🔄 Processing {len(districts)} districts × {len(years)} years...")

for district in districts:
    print(f"\n   {district}:", end=" ")
    
    district_data = climate[climate['District'] == district]
    
    for year in years:
        if year >= 2023:  # Skip incomplete years
            continue
        
        row = {'District': district, 'Year': year}
        
        # Calculate features for each growth stage
        for stage_name, stage_info in GROWTH_STAGES.items():
            stage_data = get_stage_data(district_data, year, stage_info)
            stage_features = calculate_stage_features(stage_data, stage_name)
            row.update(stage_features)
        
        # Seasonal averages (for comparison)
        season_data = district_data[
            (district_data['Year'] == year) &
            (district_data['Month'].isin([11, 12, 1, 2, 3, 4]))
        ]
        
        if len(season_data) > 0:
            row['seasonal_tmax_mean'] = season_data['Tmax'].mean()
            row['seasonal_tmin_mean'] = season_data['Tmin'].mean()
            row['seasonal_rainfall_total'] = season_data['Rainfall'].sum()
        
        all_features.append(row)
    
    years_count = len([r for r in all_features if r['District'] == district])
    print(f"✅ {years_count} years")

# Create DataFrame
features_df = pd.DataFrame(all_features)
features_df = features_df.sort_values(['District', 'Year']).reset_index(drop=True)

# Save
features_df.to_csv('data/processed/climate_features.csv', index=False)

print("\n" + "="*70)
print("✅ GROWTH STAGE FEATURES CALCULATED!")
print(f"Total rows: {len(features_df)}")
print(f"Total features: {len(features_df.columns)}")
print(f"Saved to: data/processed/climate_features.csv")
print("="*70)