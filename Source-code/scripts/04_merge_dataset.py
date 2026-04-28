"""
SCRIPT 4: MERGE FINAL DATASET (UPDATED FOR 13 DISTRICTS)
Merges climate features with yield data to create final analysis dataset

Input: climate_features.csv, yield_clean.csv
Output: wheat_final.csv (ready for analysis!)
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# CHANGE THIS TO YOUR PROJECT PATH
# ============================================================================
PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'
# ============================================================================

os.chdir(PROJECT_DIR)

print("="*70)
print("STEP 4: MERGING FINAL DATASET FOR 13 DISTRICTS")
print("="*70)

# Load data
print("\nLoading data...")
climate_features = pd.read_csv('data/processed/climate_features.csv')
yield_data = pd.read_csv('data/processed/yield_clean.csv')

print(f"✅ Climate features: {len(climate_features)} observations")
print(f"   Districts: {climate_features['District'].nunique()}")
print(f"✅ Yield data: {len(yield_data)} observations")
print(f"   Districts: {yield_data['District'].nunique()}")

# Merge on District and Year
print(f"\n{'='*70}")
print("Merging datasets...")

wheat_final = pd.merge(
    climate_features,
    yield_data,
    on=['District', 'Year'],
    how='inner'
)

print(f"✅ Merged dataset: {len(wheat_final)} observations")

# Quality checks
print(f"\n{'='*70}")
print("QUALITY CHECKS")
print(f"{'='*70}")

# Check for missing values
missing_summary = wheat_final.isnull().sum()
cols_with_missing = missing_summary[missing_summary > 0]

if len(cols_with_missing) > 0:
    print(f"\n⚠️ Columns with missing values:")
    for col, count in cols_with_missing.items():
        pct = (count / len(wheat_final)) * 100
        print(f"   {col:40} {count:4} missing ({pct:.1f}%)")
else:
    print(f"\n✅ No missing values!")

# Check for outliers in yield
print(f"\nYield statistics:")
print(f"   Mean: {wheat_final['Yield_q_ha'].mean():.2f} q/ha")
print(f"   Std: {wheat_final['Yield_q_ha'].std():.2f} q/ha")
print(f"   Min: {wheat_final['Yield_q_ha'].min():.2f} q/ha")
print(f"   Max: {wheat_final['Yield_q_ha'].max():.2f} q/ha")

# Check observations per district
print(f"\nObservations per district:")
district_counts = wheat_final.groupby('District').size().sort_values(ascending=False)
for district, count in district_counts.items():
    print(f"   {district:20} {count:3} observations")

# Expected vs actual
expected_total = wheat_final['District'].nunique() * 13
print(f"\nExpected observations: {wheat_final['District'].nunique()} districts × 13 years = {expected_total}")
print(f"Actual observations: {len(wheat_final)}")

if len(wheat_final) < expected_total:
    missing = expected_total - len(wheat_final)
    print(f"⚠️ Missing {missing} observations - some district-years incomplete")
else:
    print(f"✅ Perfect! All expected observations present!")

# Save final dataset
output_path = 'data/processed/wheat_final.csv'
wheat_final.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print("FINAL DATASET SUMMARY")
print(f"{'='*70}")
print(f"Total observations: {len(wheat_final)}")
print(f"Total districts: {wheat_final['District'].nunique()}")
print(f"Districts: {sorted(wheat_final['District'].unique())}")
print(f"Years: {wheat_final['Year'].min():.0f} to {wheat_final['Year'].max():.0f}")
print(f"Total features: {len(wheat_final.columns) - 2}")  # Exclude District, Year
print(f"   - Yield: 1 column")
print(f"   - Climate features: {len(wheat_final.columns) - 3} columns")

print(f"\nDistrict yield summary:")
for district in sorted(wheat_final['District'].unique()):
    district_data = wheat_final[wheat_final['District'] == district]
    mean_yield = district_data['Yield_q_ha'].mean()
    std_yield = district_data['Yield_q_ha'].std()
    n_obs = len(district_data)
    print(f"   {district:20} n={n_obs:2}  mean={mean_yield:5.1f}  std={std_yield:4.1f} q/ha")

print(f"\n✅ Saved to: {output_path}")
print(f"\n{'='*70}")
print("🎉 DATA PROCESSING COMPLETE!")
print(f"{'='*70}")
print(f"\n✅ Your final dataset has {len(wheat_final)} observations")
print(f"✅ Ready for analysis!")
print(f"\nNext step: Run 08_FINAL_COMPREHENSIVE_ANALYSIS.py")
print(f"{'='*70}")