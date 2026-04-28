"""
SCRIPT 1: Clean Climate Data
Run this first!

Purpose: Clean all 6 climate CSV files and combine them
Input: 6 climate CSV files in data/raw/
Output: climate_clean.csv in data/processed/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ============================================================================
# IMPORTANT: CHANGE THIS PATH TO YOUR ACTUAL FOLDER LOCATION!
# ============================================================================
# Windows example: PROJECT_DIR = r'C:\Users\John\Documents\wheat_research'
# Mac example: PROJECT_DIR = '/Users/john/Documents/wheat_research'
# Linux example: PROJECT_DIR = '/home/john/Documents/wheat_research'

PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'  # ← CHANGE THIS!

# ============================================================================

os.chdir(PROJECT_DIR)

print("="*70)
print("SCRIPT 1: CLEANING CLIMATE DATA")
print("="*70)
print(f"\nWorking directory: {os.getcwd()}")

# Climate files
climate_files = {
    'Ludhiana': 'data/raw/Ludhiana.csv',
    'Bathinda': 'data/raw/bathinda.csv',
    'Karnal': 'data/raw/Karnal.csv',
    'Meerut': 'data/raw/Meerut.csv',
    'Saharanpur': 'data/raw/Saharanpur.csv',
    'Sri_Ganganagar': 'data/raw/Sri_GangaNagur.csv',
    'Amritsar': 'data/raw/cli_amritsar.csv',
    'Patiala': 'data/raw/cli_patiala.csv',
    'Panipat': 'data/raw/cli_panipat.csv',
    'Hisar': 'data/raw/cli_hisar.csv',
    'Muzaffarnagar': 'data/raw/cli_muzz.csv',
    'Aligarh': 'data/raw/cli_aligarh.csv',
    'Hanumangarh': 'data/raw/cli_hanumangarh.csv'
}

all_climate = []

for district, filepath in climate_files.items():
    print(f"\nProcessing {district}...", end=" ")
    
    try:
        # Read file, skip metadata rows
        df = pd.read_csv(filepath, skiprows=11)
        
        # Rename columns
        df = df.rename(columns={
            'YEAR': 'Year',
            'DOY': 'DOY',
            'T2M_MAX': 'Tmax',
            'T2M_MIN': 'Tmin',
            'PRECTOTCORR': 'Rainfall'
        })
        
        # Convert DOY to Month and Day
        def doy_to_date(year, doy):
            date = datetime(int(year), 1, 1) + timedelta(int(doy) - 1)
            return date.month, date.day
        
        df['Month'], df['Day'] = zip(*df.apply(
            lambda row: doy_to_date(row['Year'], row['DOY']), axis=1
        ))
        
        # Create date
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        df['District'] = district
        
        # Handle missing values
        df['Tmax'] = df['Tmax'].replace(-999, np.nan)
        df['Tmin'] = df['Tmin'].replace(-999, np.nan)
        df['Rainfall'] = df['Rainfall'].replace(-999, np.nan)
        
        # Fill with monthly mean
        df['Tmax'] = df.groupby('Month')['Tmax'].transform(lambda x: x.fillna(x.mean()))
        df['Tmin'] = df.groupby('Month')['Tmin'].transform(lambda x: x.fillna(x.mean()))
        df['Rainfall'] = df.groupby('Month')['Rainfall'].transform(lambda x: x.fillna(x.mean()))
        
        # Filter wheat season (Nov-Apr)
        df = df[df['Month'].isin([11, 12, 1, 2, 3, 4])].copy()
        
        # Filter years 2010-2023
        df = df[(df['Year'] >= 2010) & (df['Year'] <= 2023)].copy()
        
        # Select columns
        df = df[['District', 'Date', 'Year', 'Month', 'Day', 'Tmax', 'Tmin', 'Rainfall']]
        
        all_climate.append(df)
        print(f"✅ {len(df):,} days")
        
    except Exception as e:
        print(f"❌ Error: {str(e)[:60]}")

# Combine all
climate_clean = pd.concat(all_climate, ignore_index=True)
climate_clean = climate_clean.sort_values(['District', 'Date']).reset_index(drop=True)

# Save
climate_clean.to_csv('data/processed/climate_clean.csv', index=False)

print("\n" + "="*70)
print("✅ CLIMATE DATA CLEANED!")
print(f"Total rows: {len(climate_clean):,}")
print(f"Saved to: data/processed/climate_clean.csv")
print("="*70)