"""
MACHINE LEARNING MODELING (FINAL CORRECTED VERSION)
Accounts for district baselines and temporal trends

Purpose: Test if growth stage features improve yield prediction
        AFTER controlling for district and year effects
Input: wheat_final.csv
Output: Model performance comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
PROJECT_DIR = r'C:\Users\Victus\Downloads\Wheat-research'
# ============================================================================

os.chdir(PROJECT_DIR)
os.makedirs('results', exist_ok=True)

print("="*70)
print("MACHINE LEARNING WITH DISTRICT & YEAR CONTROLS")
print("Testing: Do growth stage features add value beyond seasonal?")
print("="*70)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n📊 Loading data...")
df = pd.read_csv('data/processed/wheat_final.csv')
print(f"✅ Loaded: {len(df)} observations")

# Create dummy variables for districts and year trend
df_encoded = pd.get_dummies(df, columns=['District'], prefix='D', drop_first=True)
df_encoded['Year_trend'] = df_encoded['Year'] - 2010  # Linear time trend

# Define feature sets
control_features = [col for col in df_encoded.columns if col.startswith('D_')] + ['Year_trend']

seasonal_features = [
    'seasonal_tmax_mean',
    'seasonal_tmin_mean',
    'seasonal_rainfall_total'
]

growth_stage_features = [
    'flowering_tmax_mean', 'flowering_day_night_diff',
    'flowering_rainfall_total', 'flowering_extreme_heat_days',
    'grain_filling_tmax_mean', 'grain_filling_day_night_diff',
    'grain_filling_rainfall_total', 'grain_filling_extreme_heat_days',
    'maturation_tmax_mean', 'maturation_day_night_diff',
    'maturation_rainfall_total', 'maturation_extreme_heat_days'
]

print(f"\nFeature sets:")
print(f"   Control (District + Year): {len(control_features)} features")
print(f"   Seasonal climate: {len(seasonal_features)} features")
print(f"   Growth stage climate: {len(growth_stage_features)} features")

y = df_encoded['Yield_q_ha'].values

# ============================================================================
# 2. BUILD MODELS
# ============================================================================

# Model 0: Controls only (baseline - no climate)
# Model A: Controls + Seasonal features
# Model B: Controls + Growth stage features

cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}

print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# --------------------------
# MODEL 0: Controls Only
# --------------------------
print("\n🔄 Model 0: District + Year only (no climate)...")

X_0 = df_encoded[control_features].values
scaler_0 = StandardScaler()
X_0_scaled = scaler_0.fit_transform(X_0)

lr_0 = LinearRegression()
scores_0 = cross_val_score(lr_0, X_0_scaled, y, cv=cv, scoring='r2')
rmse_0 = -cross_val_score(lr_0, X_0_scaled, y, cv=cv, 
                          scoring='neg_root_mean_squared_error')

results['Model 0 (Controls)'] = {
    'r2_mean': scores_0.mean(),
    'r2_std': scores_0.std(),
    'rmse_mean': rmse_0.mean(),
    'rmse_std': rmse_0.std(),
    'features': control_features
}

print(f"   R² = {scores_0.mean():.4f} (±{scores_0.std():.4f})")
print(f"   RMSE = {rmse_0.mean():.2f} (±{rmse_0.std():.2f}) q/ha")

# --------------------------
# MODEL A: Controls + Seasonal
# --------------------------
print("\n🔄 Model A: District + Year + Seasonal climate...")

X_A = df_encoded[control_features + seasonal_features].values
scaler_A = StandardScaler()
X_A_scaled = scaler_A.fit_transform(X_A)

lr_A = LinearRegression()
scores_A = cross_val_score(lr_A, X_A_scaled, y, cv=cv, scoring='r2')
rmse_A = -cross_val_score(lr_A, X_A_scaled, y, cv=cv,
                          scoring='neg_root_mean_squared_error')

results['Model A (Seasonal)'] = {
    'r2_mean': scores_A.mean(),
    'r2_std': scores_A.std(),
    'rmse_mean': rmse_A.mean(),
    'rmse_std': rmse_A.std(),
    'features': control_features + seasonal_features
}

print(f"   R² = {scores_A.mean():.4f} (±{scores_A.std():.4f})")
print(f"   RMSE = {rmse_A.mean():.2f} (±{rmse_A.std():.2f}) q/ha")

climate_contribution_A = scores_A.mean() - scores_0.mean()
print(f"   📊 Climate contribution: +{climate_contribution_A:.4f} R²")

# --------------------------
# MODEL B: Controls + Growth Stage
# --------------------------
print("\n🔄 Model B: District + Year + Growth stage climate...")

X_B = df_encoded[control_features + growth_stage_features].values
scaler_B = StandardScaler()
X_B_scaled = scaler_B.fit_transform(X_B)

lr_B = LinearRegression()
scores_B = cross_val_score(lr_B, X_B_scaled, y, cv=cv, scoring='r2')
rmse_B = -cross_val_score(lr_B, X_B_scaled, y, cv=cv,
                          scoring='neg_root_mean_squared_error')

results['Model B (Growth Stage)'] = {
    'r2_mean': scores_B.mean(),
    'r2_std': scores_B.std(),
    'rmse_mean': rmse_B.mean(),
    'rmse_std': rmse_B.std(),
    'features': control_features + growth_stage_features
}

print(f"   R² = {scores_B.mean():.4f} (±{scores_B.std():.4f})")
print(f"   RMSE = {rmse_B.mean():.2f} (±{rmse_B.std():.2f}) q/ha")

climate_contribution_B = scores_B.mean() - scores_0.mean()
print(f"   📊 Climate contribution: +{climate_contribution_B:.4f} R²")

# ============================================================================
# 3. COMPARISON
# ============================================================================
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

print(f"\n📊 Model Performance:")
print(f"\n{'Model':<25} {'R²':<12} {'RMSE (q/ha)':<15} {'Climate Adds'}")
print("-" * 70)
print(f"{'0. Controls only':<25} {results['Model 0 (Controls)']['r2_mean']:>6.4f}      "
      f"{results['Model 0 (Controls)']['rmse_mean']:>6.2f}         {'(baseline)'}")
print(f"{'A. + Seasonal features':<25} {results['Model A (Seasonal)']['r2_mean']:>6.4f}      "
      f"{results['Model A (Seasonal)']['rmse_mean']:>6.2f}         {climate_contribution_A:>+.4f}")
print(f"{'B. + Growth stage features':<25} {results['Model B (Growth Stage)']['r2_mean']:>6.4f}      "
      f"{results['Model B (Growth Stage)']['rmse_mean']:>6.2f}         {climate_contribution_B:>+.4f}")

# Calculate improvement of B over A
if results['Model A (Seasonal)']['r2_mean'] > 0:
    improvement_pct = ((results['Model B (Growth Stage)']['r2_mean'] - 
                       results['Model A (Seasonal)']['r2_mean']) / 
                       results['Model A (Seasonal)']['r2_mean']) * 100
else:
    improvement_pct = 0

delta_r2 = results['Model B (Growth Stage)']['r2_mean'] - results['Model A (Seasonal)']['r2_mean']
delta_rmse = results['Model A (Seasonal)']['rmse_mean'] - results['Model B (Growth Stage)']['rmse_mean']

print(f"\n🎯 KEY FINDING:")
print(f"   Growth stage features add {delta_r2:+.4f} R² beyond seasonal")
print(f"   Growth stage features reduce RMSE by {delta_rmse:+.2f} q/ha")

if delta_r2 > 0.02:
    print(f"   ✅ VALIDATED: Growth stage features provide meaningful improvement!")
elif delta_r2 > 0:
    print(f"   ✅ POSITIVE: Growth stage features show slight improvement")
else:
    print(f"   ⚠️  No improvement detected")

# ============================================================================
# 4. TRAIN FINAL MODEL FOR FEATURE IMPORTANCE
# ============================================================================
print("\n📊 Training Random Forest for feature interpretation...")

# Train on full data
rf_B = RandomForestRegressor(n_estimators=100, max_depth=5, 
                             min_samples_split=5, random_state=42)
rf_B.fit(X_B, y)

# Get feature importance (climate features only, excluding controls)
all_features_B = control_features + growth_stage_features
feature_importance = pd.DataFrame({
    'feature': all_features_B,
    'importance': rf_B.feature_importances_
})

# Separate climate features
climate_importance = feature_importance[
    ~feature_importance['feature'].isin(control_features)
].sort_values('importance', ascending=False)

print(f"\n📊 Top climate features (excluding district/year controls):")
for i, row in climate_importance.head(8).iterrows():
    print(f"   {row['feature']:<35} {row['importance']:.4f}")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n📊 Creating visualizations...")

# Figure 1: Model Comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Controls\nOnly', 'Controls +\nSeasonal', 'Controls +\nGrowth Stage']
r2_values = [results['Model 0 (Controls)']['r2_mean'],
             results['Model A (Seasonal)']['r2_mean'],
             results['Model B (Growth Stage)']['r2_mean']]
r2_errors = [results['Model 0 (Controls)']['r2_std'],
             results['Model A (Seasonal)']['r2_std'],
             results['Model B (Growth Stage)']['r2_std']]

colors = ['gray', 'skyblue', 'coral']
bars = ax.bar(models, r2_values, yerr=r2_errors, color=colors, 
              edgecolor='black', linewidth=1.5, capsize=5)

ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance: Effect of Climate Features\n(with District & Year Controls)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(bottom=0)

# Add value labels
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add improvement arrows
if delta_r2 > 0:
    ax.annotate('', xy=(2, results['Model B (Growth Stage)']['r2_mean']), 
                xytext=(1, results['Model A (Seasonal)']['r2_mean']),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    mid_y = (results['Model A (Seasonal)']['r2_mean'] + 
             results['Model B (Growth Stage)']['r2_mean']) / 2
    ax.text(1.5, mid_y, f'+{delta_r2:.3f}', fontsize=10, 
            color='green', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('results/13_model_comparison_final.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/13_model_comparison_final.png")
plt.close()

# Figure 2: Climate Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))

top_climate = climate_importance.head(12)
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(top_climate)))

ax.barh(range(len(top_climate)), top_climate['importance'], 
        color=colors_imp, edgecolor='black')
ax.set_yticks(range(len(top_climate)))
ax.set_yticklabels(top_climate['feature'])
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('Top Climate Features (Random Forest)',
             fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/14_climate_feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: results/14_climate_feature_importance.png")
plt.close()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n💾 Saving results...")

# Summary table
summary_df = pd.DataFrame({
    'Model': ['Controls Only', 'Controls + Seasonal', 'Controls + Growth Stage'],
    'R²': [r2_values[0], r2_values[1], r2_values[2]],
    'R² Std': [r2_errors[0], r2_errors[1], r2_errors[2]],
    'RMSE': [results['Model 0 (Controls)']['rmse_mean'],
             results['Model A (Seasonal)']['rmse_mean'],
             results['Model B (Growth Stage)']['rmse_mean']],
    'Climate Contribution': [0, climate_contribution_A, climate_contribution_B]
})

summary_df.to_csv('results/model_summary_final.csv', index=False)
climate_importance.to_csv('results/climate_features_importance.csv', index=False)

print("   ✅ Saved: results/model_summary_final.csv")
print("   ✅ Saved: results/climate_features_importance.csv")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 ANALYSIS COMPLETE!")
print("="*70)

print(f"\n📊 RESULTS SUMMARY:")
print(f"\n1. BASELINE (District + Year controls):")
print(f"   R² = {results['Model 0 (Controls)']['r2_mean']:.4f}")
print(f"   → Explains {results['Model 0 (Controls)']['r2_mean']*100:.1f}% of yield variation")

print(f"\n2. ADDING SEASONAL CLIMATE:")
print(f"   R² = {results['Model A (Seasonal)']['r2_mean']:.4f}")
print(f"   → Climate adds {climate_contribution_A:.4f} R² ({climate_contribution_A*100:.1f}%)")

print(f"\n3. ADDING GROWTH STAGE CLIMATE:")
print(f"   R² = {results['Model B (Growth Stage)']['r2_mean']:.4f}")
print(f"   → Climate adds {climate_contribution_B:.4f} R² ({climate_contribution_B*100:.1f}%)")

print(f"\n🎯 HYPOTHESIS TEST:")
print(f"   Growth stage vs Seasonal: {delta_r2:+.4f} R²")

if delta_r2 > 0.03:
    print(f"   ✅ STRONG EVIDENCE: Growth stage features substantially better!")
    print(f"   Your research contribution is clearly demonstrated.")
elif delta_r2 > 0.01:
    print(f"   ✅ POSITIVE EVIDENCE: Growth stage features improve predictions")
    print(f"   Your hypothesis has support.")
elif delta_r2 > 0:
    print(f"   ✅ SLIGHT IMPROVEMENT: Growth stage shows promise")
else:
    print(f"   ⚠️  No clear improvement detected")
    print(f"   Both approaches perform similarly.")

print(f"\n💡 INTERPRETATION:")
if results['Model 0 (Controls)']['r2_mean'] > 0.6:
    print(f"   • District and year explain most variation (R²={results['Model 0 (Controls)']['r2_mean']:.2f})")
    print(f"   • Climate adds modest but important predictive power")
    
if delta_r2 > 0:
    print(f"   • Growth stage features capture {delta_r2*100:.1f}% more variation than seasonal")
    print(f"   • Temporal disaggregation provides value!")

print("\n" + "="*70)