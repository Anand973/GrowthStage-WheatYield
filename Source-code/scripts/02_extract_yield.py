"""
SCRIPT 2: EXTRACT YIELD DATA (FIXED FOR WIDE FORMAT)
Handles yield files where years are COLUMNS (wide format)

Format: Each year is a column like "2010-2011", "2011-2012", etc.
Data is in Ton/Ha - converts to q/ha (×10)
"""

import pandas as pd
import numpy as np
import os
import re

# ============================================================================
# CHANGE THIS TO YOUR PROJECT PATH
# ============================================================================
PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'
# ============================================================================

os.chdir(PROJECT_DIR)
os.makedirs('data/processed', exist_ok=True)

print("="*70)
print("STEP 2: EXTRACTING YIELD DATA FOR 13 DISTRICTS")
print("="*70)

# Define all 13 district yield files
yield_files = {
    'Ludhiana': 'data/raw/crop_2ludhiana.xls',
    'Bathinda': 'data/raw/crop_2bathinda.xls',
    'Karnal': 'data/raw/crop_2karnal.xls',
    'Saharanpur': 'data/raw/crop_sharanpur.xls',
    'Meerut': 'data/raw/crop_2meerut.xls',
    'Sri_Ganganagar': 'data/raw/crop_sri-ganagnagur.xls',
    
    # New 7 districts
    'Amritsar': 'data/raw/crop_amritsar.xls',
    'Patiala': 'data/raw/crop_patiala.xls',
    'Panipat': 'data/raw/crop_panipat.xls',
    'Hisar': 'data/raw/crop_hisar.xls',
    'Muzaffarnagar': 'data/raw/crop_muzz.xls',
    'Aligarh': 'data/raw/crop_aligarh.xls',
    'Hanumangarh': 'data/raw/crop_hanumangarh.xls'
}

def extract_year_from_column(col_name):
    """Extract starting year from column names like '2010 - 2011'"""
    if isinstance(col_name, tuple):
        col_name = str(col_name[0])  # Handle multi-level headers
    else:
        col_name = str(col_name)
    
    # Extract first 4-digit year
    match = re.search(r'20\d{2}', col_name)
    if match:
        return int(match.group())
    return None

all_yield_data = []

print(f"\nProcessing {len(yield_files)} districts...")

for district, filepath in yield_files.items():
    print(f"\n{'='*70}")
    print(f"Processing: {district}")
    print(f"File: {filepath}")
    
    try:
        # Read Excel file - this gets the table structure
        tables = pd.read_html(filepath)
        
        if len(tables) == 0:
            print(f"   ❌ No tables found in file")
            continue
        
        df = tables[0]  # Get first table
        
        print(f"   File shape: {df.shape}")
        print(f"   Sample columns: {df.columns.tolist()[:5]}")
        
        # The data is in WIDE FORMAT: one row, multiple year columns
        # Columns are like: ('2010 - 2011', 'Yield (Ton./Ha.)')
        
        # Extract yield data by finding columns with years
        year_data = []
        
        for col in df.columns:
            year = extract_year_from_column(col)
            
            if year is not None and 2010 <= year <= 2022:
                # Get the yield value from this column
                # Data is in first row (index 0)
                try:
                    yield_value = df[col].iloc[0]
                    
                    # Convert to numeric
                    if pd.notna(yield_value):
                        yield_numeric = pd.to_numeric(yield_value, errors='coerce')
                        
                        if pd.notna(yield_numeric) and yield_numeric > 0:
                            # Convert from Ton/Ha to q/ha (1 ton = 10 quintals)
                            yield_q_ha = yield_numeric * 10
                            
                            year_data.append({
                                'District': district,
                                'Year': year,
                                'Yield_q_ha': yield_q_ha
                            })
                except Exception as e:
                    continue
        
        if len(year_data) > 0:
            district_df = pd.DataFrame(year_data)
            all_yield_data.append(district_df)
            
            print(f"   ✅ Extracted {len(district_df)} years of yield data")
            print(f"   Years: {district_df['Year'].min():.0f} to {district_df['Year'].max():.0f}")
            print(f"   Yield range: {district_df['Yield_q_ha'].min():.1f} to {district_df['Yield_q_ha'].max():.1f} q/ha")
        else:
            print(f"   ⚠️ No valid yield data found")
        
    except Exception as e:
        print(f"   ❌ ERROR processing {district}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Combine all districts
print(f"\n{'='*70}")
print("Combining all districts...")

if len(all_yield_data) == 0:
    print("❌ ERROR: No yield data extracted from any district!")
    print("Please check the yield file formats.")
    exit(1)

yield_clean = pd.concat(all_yield_data, ignore_index=True)

# Sort
yield_clean = yield_clean.sort_values(['District', 'Year'])

# Save
output_path = 'data/processed/yield_clean.csv'
yield_clean.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"Total districts: {yield_clean['District'].nunique()}")
print(f"Districts: {sorted(yield_clean['District'].unique())}")
print(f"Total observations: {len(yield_clean)}")
print(f"Year range: {yield_clean['Year'].min():.0f} to {yield_clean['Year'].max():.0f}")
print(f"\nYield by district:")
for district in sorted(yield_clean['District'].unique()):
    district_yields = yield_clean[yield_clean['District'] == district]
    mean_yield = district_yields['Yield_q_ha'].mean()
    n_years = len(district_yields)
    years_list = sorted(district_yields['Year'].unique())
    print(f"   {district:20} {n_years:2} years, mean = {mean_yield:5.1f} q/ha")
    print(f"                        Years: {years_list}")

print(f"\n✅ Saved to: {output_path}")
print(f"\nExpected for final dataset: {yield_clean['District'].nunique()} districts × 13 years = {yield_clean['District'].nunique() * 13} observations")
print(f"Actual observations: {len(yield_clean)}")

if len(yield_clean) < yield_clean['District'].nunique() * 13:
    missing = (yield_clean['District'].nunique() * 13) - len(yield_clean)
    print(f"\n⚠️ WARNING: Missing {missing} observations - some district-years incomplete")
    print(f"This is OK - analysis will work with available data")
else:
    print(f"\n✅ Perfect! All years present for all districts.")

print(f"{'='*70}")