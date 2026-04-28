"""
EXPLORATORY DATA ANALYSIS (EDA)
Complete analysis of wheat yield dataset

Purpose: Understand patterns, relationships, and validate research hypothesis
Input: wheat_final.csv
Output: Multiple visualizations in results/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("hls")

# ============================================================================
# CHANGE THIS TO YOUR PROJECT PATH
# ============================================================================
PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'  # Your actual path
# ============================================================================

os.chdir(PROJECT_DIR)

# Create results folder
os.makedirs('results', exist_ok=True)

print("="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# Load data
df = pd.read_csv('data/processed/wheat_final.csv')

print(f"\n✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
print(f"Districts: {df['District'].nunique()}")
print(f"Years: {df['Year'].min()} to {df['Year'].max()}")

# ============================================================================
# 1. YIELD TRENDS OVER TIME BY DISTRICT
# ============================================================================
print("\n📊 Creating Figure 1: Yield Trends by District...")

plt.figure(figsize=(14, 8))

districts = df['District'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(districts)))

for i, district in enumerate(sorted(districts)):
    district_data = df[df['District'] == district].sort_values('Year')
    plt.plot(district_data['Year'], district_data['Yield_q_ha'], 
             marker='o', linewidth=2, markersize=8,
             label=district, color=colors[i])

plt.xlabel('Year', fontsize=14, fontweight='bold')
plt.ylabel('Yield (quintals/ha)', fontsize=14, fontweight='bold')
plt.title('Wheat Yield Trends by District (2010-2022)', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.xticks(range(2010, 2023, 2))

# Highlight 2022 heatwave
plt.axvline(x=2022, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.text(2022, plt.ylim()[1]*0.95, '2022 Heatwave', 
         ha='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('results/01_yield_trends.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/01_yield_trends.png")
plt.close()

# ============================================================================
# 2. DISTRICT COMPARISON (BOXPLOT)
# ============================================================================
print("\n📊 Creating Figure 2: District Yield Comparison...")

plt.figure(figsize=(12, 7))

# Sort districts by median yield
district_order = df.groupby('District')['Yield_q_ha'].median().sort_values(ascending=False).index

sns.boxplot(data=df, x='District', y='Yield_q_ha', order=district_order,
            palette='Set2', width=0.6)
sns.swarmplot(data=df, x='District', y='Yield_q_ha', order=district_order,
              color='black', alpha=0.5, size=4)

plt.xlabel('District', fontsize=14, fontweight='bold')
plt.ylabel('Yield (quintals/ha)', fontsize=14, fontweight='bold')
plt.title('Yield Distribution by District (2010-2022)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)

# Add mean line
means = df.groupby('District')['Yield_q_ha'].mean()
plt.plot(range(len(district_order)), [means[d] for d in district_order], 
         'r--', linewidth=2, label='Mean', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('results/02_district_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/02_district_comparison.png")
plt.close()

# ============================================================================
# 3. CORRELATION HEATMAP (TOP FEATURES)
# ============================================================================
print("\n📊 Creating Figure 3: Correlation Heatmap...")

# Select key features for visualization
key_features = [
    'Yield_q_ha',
    'flowering_tmax_mean',
    'flowering_rainfall_total',
    'flowering_extreme_heat_days',
    'grain_filling_tmax_mean',
    'grain_filling_rainfall_total',
    'grain_filling_extreme_heat_days',
    'maturation_tmax_mean',
    'maturation_rainfall_total',
    'seasonal_tmax_mean',
    'seasonal_rainfall_total'
]

plt.figure(figsize=(12, 10))
corr_matrix = df[key_features].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1,
            cbar_kws={'label': 'Correlation'})

plt.title('Correlation Matrix: Key Climate Features vs Yield', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('results/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/03_correlation_heatmap.png")
plt.close()

# ============================================================================
# 4. TOP 15 FEATURES CORRELATED WITH YIELD
# ============================================================================
print("\n📊 Creating Figure 4: Top Features Correlation...")

# Calculate correlations
feature_cols = [col for col in df.columns if col not in ['District', 'Year', 'Yield_q_ha']]
correlations = df[feature_cols].corrwith(df['Yield_q_ha']).sort_values(ascending=False)

# Top 15
top_15 = correlations.head(15)

plt.figure(figsize=(10, 8))
colors_bar = ['green' if x > 0 else 'red' for x in top_15.values]
top_15.plot(kind='barh', color=colors_bar, edgecolor='black', linewidth=0.7)

plt.xlabel('Correlation with Yield', fontsize=12, fontweight='bold')
plt.ylabel('Climate Feature', fontsize=12, fontweight='bold')
plt.title('Top 15 Features Correlated with Wheat Yield', 
          fontsize=14, fontweight='bold', pad=15)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/04_top_features.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/04_top_features.png")
plt.close()

# ============================================================================
# 5. GROWTH STAGE COMPARISON
# ============================================================================
print("\n📊 Creating Figure 5: Growth Stage Importance...")

# Average correlation by growth stage
flowering_cols = [col for col in df.columns if 'flowering' in col and col != 'Yield_q_ha']
grain_cols = [col for col in df.columns if 'grain_filling' in col and col != 'Yield_q_ha']
maturation_cols = [col for col in df.columns if 'maturation' in col and col != 'Yield_q_ha']
seasonal_cols = [col for col in df.columns if 'seasonal' in col and col != 'Yield_q_ha']

stage_correlations = {
    'Flowering\n(Feb 1-15)': df[flowering_cols].corrwith(df['Yield_q_ha']).abs().mean(),
    'Grain Filling\n(Feb 16-Mar 15)': df[grain_cols].corrwith(df['Yield_q_ha']).abs().mean(),
    'Maturation\n(Mar 16-Apr 10)': df[maturation_cols].corrwith(df['Yield_q_ha']).abs().mean(),
    'Seasonal\n(Nov-Apr)': df[seasonal_cols].corrwith(df['Yield_q_ha']).abs().mean()
}

plt.figure(figsize=(10, 6))
stages = list(stage_correlations.keys())
values = list(stage_correlations.values())
colors_stage = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = plt.bar(stages, values, color=colors_stage, edgecolor='black', linewidth=1.5)

plt.ylabel('Average Absolute Correlation', fontsize=12, fontweight='bold')
plt.title('Average Feature Importance by Growth Stage', 
          fontsize=14, fontweight='bold', pad=15)
plt.ylim(0, max(values) * 1.2)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/05_growth_stage_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/05_growth_stage_comparison.png")
plt.close()

# ============================================================================
# 6. 2022 HEATWAVE ANALYSIS
# ============================================================================
print("\n📊 Creating Figure 6: 2022 Heatwave Impact...")

# Compare 2022 vs average
df_2022 = df[df['Year'] == 2022]
df_other = df[df['Year'] != 2022]

comparison_data = []
for district in sorted(df['District'].unique()):
    avg_yield = df_other[df_other['District'] == district]['Yield_q_ha'].mean()
    yield_2022 = df_2022[df_2022['District'] == district]['Yield_q_ha'].values[0]
    
    comparison_data.append({
        'District': district,
        'Average (2010-2021)': avg_yield,
        '2022 (Heatwave)': yield_2022,
        'Difference': yield_2022 - avg_yield,
        'Percent Change': ((yield_2022 - avg_yield) / avg_yield) * 100
    })

comp_df = pd.DataFrame(comparison_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Absolute yields
x = np.arange(len(comp_df))
width = 0.35

ax1.bar(x - width/2, comp_df['Average (2010-2021)'], width, 
        label='Average (2010-2021)', color='skyblue', edgecolor='black')
ax1.bar(x + width/2, comp_df['2022 (Heatwave)'], width, 
        label='2022 (Heatwave)', color='coral', edgecolor='black')

ax1.set_xlabel('District', fontsize=12, fontweight='bold')
ax1.set_ylabel('Yield (quintals/ha)', fontsize=12, fontweight='bold')
ax1.set_title('Yield Comparison: 2022 vs Historical Average', 
              fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(comp_df['District'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)

# Percent change
colors_change = ['red' if x < 0 else 'green' for x in comp_df['Percent Change']]
ax2.barh(comp_df['District'], comp_df['Percent Change'], 
         color=colors_change, edgecolor='black')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('Yield Change (%)', fontsize=12, fontweight='bold')
ax2.set_title('2022 Yield Change from Average (%)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/06_heatwave_impact.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/06_heatwave_impact.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("📊 SUMMARY STATISTICS")
print("="*70)

print("\n1. YIELD STATISTICS BY DISTRICT:")
print(df.groupby('District')['Yield_q_ha'].describe().round(2))

print("\n2. TOP 10 FEATURES CORRELATED WITH YIELD:")
print(correlations.head(10))

print("\n3. 2022 HEATWAVE IMPACT:")
print(comp_df[['District', 'Average (2010-2021)', '2022 (Heatwave)', 'Percent Change']].to_string(index=False))

print("\n4. GROWTH STAGE IMPORTANCE:")
for stage, value in stage_correlations.items():
    print(f"   {stage.replace(chr(10), ' ')}: {value:.4f}")

print("\n" + "="*70)
print("✅ EDA COMPLETE!")
print("="*70)
print(f"\n📁 All visualizations saved in: {PROJECT_DIR}/results/")
print("\nGenerated 6 figures:")
print("   1. Yield trends by district")
print("   2. District yield comparison")
print("   3. Correlation heatmap")
print("   4. Top 15 features")
print("   5. Growth stage comparison")
print("   6. 2022 heatwave impact")
print("\n🎯 Next step: Build ML models to test your hypothesis!")
print("="*70)