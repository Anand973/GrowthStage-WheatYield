

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# CHANGE THIS TO YOUR PROJECT PATH
# ============================================================================
PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'
# ============================================================================

os.chdir(PROJECT_DIR)

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("="*70)
print("GENERATING ALL FIGURES FOR RESEARCH PAPER")
print("="*70)

# Load results from comprehensive analysis
try:
    seasonal_corr = pd.read_csv('results/seasonal_features_correlations.csv')
    growth_corr = pd.read_csv('results/growth_stage_features_correlations.csv')
    print("\n✅ Loaded correlation data")
except:
    print("\n❌ ERROR: Run 08_FINAL_COMPREHENSIVE_ANALYSIS.py first!")
    exit(1)

# ============================================================================
# FIGURE 1: MODEL PERFORMANCE COMPARISON (BAR CHART)
# ============================================================================

print("\nCreating Figure 1: Model Performance Comparison...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Your actual results
metrics = ['R²', 'Adjusted R²', 'RMSE (q/ha)', 'MAE (q/ha)']
model_a = [0.040, 0.023, 4.09, 3.16]
model_b = [0.199, 0.138, 3.74, 2.88]

# R² Comparison
ax1.bar(['Model A\n(Seasonal)', 'Model B\n(Growth Stage)'], [model_a[0], model_b[0]], 
        color=['#E74C3C', '#27AE60'], alpha=0.8, edgecolor='black')
ax1.set_ylabel('R² (Variance Explained)', fontweight='bold')
ax1.set_title('(a) R² Comparison', fontweight='bold')
ax1.set_ylim(0, 0.25)
for i, v in enumerate([model_a[0], model_b[0]]):
    ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
ax1.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% threshold')
ax1.grid(axis='y', alpha=0.3)

# Adjusted R² Comparison
ax2.bar(['Model A', 'Model B'], [model_a[1], model_b[1]], 
        color=['#E74C3C', '#27AE60'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Adjusted R²', fontweight='bold')
ax2.set_title('(b) Adjusted R² Comparison', fontweight='bold')
ax2.set_ylim(0, 0.18)
for i, v in enumerate([model_a[1], model_b[1]]):
    ax2.text(i, v + 0.008, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# RMSE Comparison
ax3.bar(['Model A', 'Model B'], [model_a[2], model_b[2]], 
        color=['#E74C3C', '#27AE60'], alpha=0.8, edgecolor='black')
ax3.set_ylabel('RMSE (q/ha)', fontweight='bold')
ax3.set_title('(c) RMSE Comparison (Lower is Better)', fontweight='bold')
ax3.set_ylim(0, 5)
for i, v in enumerate([model_a[2], model_b[2]]):
    ax3.text(i, v + 0.15, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)
ax3.grid(axis='y', alpha=0.3)

# MAE Comparison
ax4.bar(['Model A', 'Model B'], [model_a[3], model_b[3]], 
        color=['#E74C3C', '#27AE60'], alpha=0.8, edgecolor='black')
ax4.set_ylabel('MAE (q/ha)', fontweight='bold')
ax4.set_title('(d) MAE Comparison (Lower is Better)', fontweight='bold')
ax4.set_ylim(0, 4)
for i, v in enumerate([model_a[3], model_b[3]]):
    ax4.text(i, v + 0.12, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Figure 1: Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/Figure_1_Model_Performance.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: results/Figure_1_Model_Performance.png")
plt.close()

# ============================================================================
# FIGURE 2: CORRELATION COMPARISON (ALL FEATURES)
# ============================================================================

print("\nCreating Figure 2: Correlation Comparison...")

fig, ax = plt.subplots(figsize=(14, 6))

# Prepare data
all_features = []
all_corrs = []
all_pvals = []
all_types = []

# Seasonal features
for _, row in seasonal_corr.iterrows():
    all_features.append(row['Feature'].replace('seasonal_', 'Seasonal: '))
    all_corrs.append(row['Correlation'])
    all_pvals.append(row['P_value'])
    all_types.append('Seasonal')

# Growth stage features
for _, row in growth_corr.iterrows():
    all_features.append(row['Feature'].replace('_', ' ').title())
    all_corrs.append(row['Correlation'])
    all_pvals.append(row['P_value'])
    all_types.append('Growth Stage')

# Create dataframe
plot_df = pd.DataFrame({
    'Feature': all_features,
    'Correlation': all_corrs,
    'P_value': all_pvals,
    'Type': all_types,
    'Significant': ['Yes' if p < 0.05 else 'No' for p in all_pvals]
})

# Sort by absolute correlation
plot_df['Abs_Corr'] = plot_df['Correlation'].abs()
plot_df = plot_df.sort_values('Abs_Corr', ascending=True)

# Plot
colors = ['#27AE60' if sig == 'Yes' else '#BDC3C7' for sig in plot_df['Significant']]
bars = ax.barh(range(len(plot_df)), plot_df['Correlation'], color=colors, alpha=0.8, edgecolor='black')

# Add significance markers
for i, (corr, pval) in enumerate(zip(plot_df['Correlation'], plot_df['P_value'])):
    if pval < 0.01:
        marker = '**'
    elif pval < 0.05:
        marker = '*'
    else:
        marker = ''
    
    if marker:
        x_pos = corr + 0.01 if corr > 0 else corr - 0.01
        ax.text(x_pos, i, marker, va='center', ha='left' if corr > 0 else 'right', 
                fontsize=12, fontweight='bold', color='darkgreen')

ax.set_yticks(range(len(plot_df)))
ax.set_yticklabels(plot_df['Feature'], fontsize=8)
ax.set_xlabel('Correlation Coefficient (r)', fontweight='bold', fontsize=11)
ax.set_title('Figure 2: Correlation Strength Comparison\n(Green = Significant at p<0.05; * p<0.05, ** p<0.01)', 
             fontweight='bold', fontsize=12, pad=15)
ax.axvline(x=0, color='black', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27AE60', label='Significant (p < 0.05)'),
    Patch(facecolor='#BDC3C7', label='Not Significant')
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('results/Figure_2_Correlation_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: results/Figure_2_Correlation_Comparison.png")
plt.close()

# ============================================================================
# FIGURE 3: GROWTH STAGE CONTRIBUTIONS
# ============================================================================

print("\nCreating Figure 3: Growth Stage Contributions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Calculate average correlation by growth stage
growth_stage_avg = growth_corr.copy()
growth_stage_avg['Stage'] = growth_stage_avg['Feature'].apply(
    lambda x: 'Flowering' if 'flowering' in x else 
              'Grain Filling' if 'grain' in x else 
              'Maturation'
)

stage_stats = growth_stage_avg.groupby('Stage')['Abs_Correlation'].agg(['mean', 'std', 'count'])
stage_stats = stage_stats.reindex(['Flowering', 'Grain Filling', 'Maturation'])

# Plot 1: Average correlation by stage
bars1 = ax1.bar(stage_stats.index, stage_stats['mean'], 
                color=['#3498DB', '#E67E22', '#E74C3C'], alpha=0.8, edgecolor='black',
                yerr=stage_stats['std'], capsize=5, error_kw={'linewidth': 2})
ax1.set_ylabel('Average |Correlation|', fontweight='bold', fontsize=11)
ax1.set_xlabel('Growth Stage', fontweight='bold', fontsize=11)
ax1.set_title('(a) Average Correlation Strength by Growth Stage', fontweight='bold', fontsize=11)
ax1.set_ylim(0, 0.20)

# Add value labels
for i, (idx, row) in enumerate(stage_stats.iterrows()):
    ax1.text(i, row['mean'] + row['std'] + 0.01, f"{row['mean']:.3f}", 
             ha='center', fontweight='bold', fontsize=10)

ax1.grid(axis='y', alpha=0.3)

# Plot 2: Number of significant features by stage
sig_counts = growth_stage_avg[growth_stage_avg['Significant'] == 'Yes'].groupby('Stage').size()
sig_counts = sig_counts.reindex(['Flowering', 'Grain Filling', 'Maturation'], fill_value=0)

bars2 = ax2.bar(sig_counts.index, sig_counts.values, 
                color=['#3498DB', '#E67E22', '#E74C3C'], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Number of Significant Features', fontweight='bold', fontsize=11)
ax2.set_xlabel('Growth Stage', fontweight='bold', fontsize=11)
ax2.set_title('(b) Significant Features by Growth Stage (p<0.05)', fontweight='bold', fontsize=11)
ax2.set_ylim(0, 5)

# Add value labels
for i, (idx, val) in enumerate(sig_counts.items()):
    ax2.text(i, val + 0.1, str(val), ha='center', fontweight='bold', fontsize=12)

ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Figure 3: Growth Stage Climate Contributions to Yield Prediction', 
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/Figure_3_Growth_Stage_Contributions.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Created: results/Figure_3_Growth_Stage_Contributions.png")
plt.close()

print("\n" + "="*70)
print("✅ ALL FIGURES CREATED SUCCESSFULLY!")
print("="*70)
print("\nFiles created in results/ folder:")
print("  1. Figure_1_Model_Performance.png")
print("  2. Figure_2_Correlation_Comparison.png")
print("  3. Figure_3_Growth_Stage_Contributions.png")
print("\nThese are publication-quality (300 DPI) images ready to insert!")
print("="*70)