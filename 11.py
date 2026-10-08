import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Dataset
file_path ='Iris.csv'
df = pd.read_csv(file_path)

if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# 2. Setup Figure Layout
fig = plt.figure(figsize=(12, 10))

# --- SUMMARY STATISTICS TEXT SECTION (Top Half) ---
stats_summary = df.describe().round(2).to_string()
species_means = df.groupby('Species').mean().round(2).to_string()

full_text = f"=== OVERALL SUMMARY STATISTICS ===\n\n{stats_summary}\n\n" \
            f"=== MEAN VALUES BY SPECIES ===\n\n{species_means}"

plt.subplot(2, 1, 1)
plt.text(0.01, 0.95, full_text, fontsize=9, family='monospace', verticalalignment='top')
plt.axis('off')  # Hide axis borders for text area
plt.title('WEEK 11: SUMMARY STATISTICS & DATA ANALYSIS', fontsize=12, fontweight='bold')

# --- VISUALIZATIONS SECTION (Bottom Half) ---
# Scatter Plot
plt.subplot(2, 2, 3)
for species, group in df.groupby('Species'):
    plt.scatter(group['PetalLengthCm'], group['PetalWidthCm'], label=species)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Width')
plt.legend(fontsize=8)

# Feature Boxplots
plt.subplot(2, 2, 4)
df.boxplot()
plt.title('Feature Boxplots')
plt.xticks(rotation=15, fontsize=8)

plt.tight_layout()
plt.show()

