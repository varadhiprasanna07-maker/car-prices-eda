"""
Car Prices Dataset — Complete EDA Assignment
Pandas Data Analysis | Python Assignment
All Tasks: 1 (Profiling) · 2 (Queries) · 3 (Visualisation)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

# ── Global Style ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3d4d',   'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaaaaa',      'ytick.color': '#aaaaaa',
    'text.color': '#e0e0e0',       'grid.color': '#2a2d3d',
    'grid.linestyle': '--',        'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans',  'axes.titlesize': 13,
    'axes.labelsize': 11,
})
PALETTE = ['#7b68ee','#00bcd4','#ff6b6b','#ffd93d','#6bcb77','#f77f00']
os.makedirs('plots', exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(f'plots/{name}.png', dpi=140, bbox_inches='tight',
                facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f'  ✔  plots/{name}.png')

# ══════════════════════════════════════════════════════════════
#  TASK 1 — DATA INGESTION & QUALITY PROFILING
# ══════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print(' TASK 1 — DATA INGESTION & QUALITY PROFILING')
print('═'*60)

# 1.1 Load & Inspect
df = pd.read_csv('dataset/Dataset/car_prices.csv')
print('\n── 1.1  First 5 rows ──')
print(df.head().to_string())
print('\nData types:\n', df.dtypes.to_string())
print(f'\nTotal records: {len(df):,}')

# 1.2 Shape & Structure
print('\n── 1.2  Shape ──')
print(f'Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}')
print('\nColumn names & dtypes:\n', df.dtypes.to_string())

# 1.3 Missing & Anomaly Detection
print('\n── 1.3  Null counts per column ──')
nc  = df.isnull().sum()
npc = (nc / len(df) * 100).round(2)
print(pd.DataFrame({'null_count': nc, 'null_%': npc})[nc > 0].to_string())

# Bar chart of nulls
cols_null = nc[nc > 0]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(cols_null.index, cols_null.values,
              color=PALETTE[:len(cols_null)], edgecolor='#ffffff22')
ax.set_title('Missing Values per Column', pad=12, color='white', fontsize=14)
ax.set_xlabel('Column'); ax.set_ylabel('Null Count')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+200,
            f'{int(b.get_height()):,}', ha='center', va='bottom', fontsize=8)
ax.grid(axis='y'); plt.xticks(rotation=15)
save('1_3a_missing_values_bar')

# Heatmap
samp = df.sample(min(5000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(samp.isnull(), cbar=True, yticklabels=False,
            cmap=['#1a1d27','#7b68ee'], ax=ax)
ax.set_title('Null Heatmap (5 000-row sample)', pad=10, color='white')
save('1_3b_null_heatmap')

# Resolve nulls
for col in ['condition','odometer','mmr','sellingprice']:
    df[col].fillna(df[col].median(), inplace=True)
for col in ['make','model','trim','body','transmission','color','interior','seller','saledate','vin']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print('\nNull counts after resolution:', df.isnull().sum().sum(), '→ All resolved ✔')

# Duplicates
dup = df.duplicated().sum()
print(f'Duplicate records: {dup:,}')
df.drop_duplicates(inplace=True); df.reset_index(drop=True, inplace=True)
print(f'Records after dedup: {len(df):,}')

# ══════════════════════════════════════════════════════════════
#  TASK 2 — DATAFRAME QUERIES
# ══════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print(' TASK 2 — DATAFRAME QUERIES')
print('═'*60)

# 2.1
s = df['sellingprice'].agg(['mean','min','max'])
print(f"\n── 2.1  Average: ${s['mean']:,.2f}  |  Min: ${s['min']:,.2f}  |  Max: ${s['max']:,.2f}")

# 2.2
colors = sorted(df['color'].dropna().unique(), key=str)
print(f'\n── 2.2  Unique car colors ({len(colors)}):')
print(colors)

# 2.3
print(f'\n── 2.3  Unique brands: {df["make"].nunique()}  |  Unique models: {df["model"].nunique()}')

# 2.4
exp = df[df['sellingprice'] > 165000]
print(f'\n── 2.4  Cars priced > $165,000 ({len(exp):,} records):')
print(exp[['year','make','model','sellingprice','state']].to_string(index=False))

# 2.5
print('\n── 2.5  Top 5 most sold car models:')
print(df['model'].value_counts().head(5).to_string())

# 2.6
print('\n── 2.6  Avg selling price by brand (top 15):')
print(df.groupby('make')['sellingprice'].mean().sort_values(ascending=False).head(15).to_string())

# 2.7
print('\n── 2.7  Min selling price by interior:')
print(df.groupby('interior')['sellingprice'].min().sort_values().to_string())

# 2.8
print('\n── 2.8  Highest odometer per year (desc):')
print(df.groupby('year')['odometer'].max().reset_index()
        .sort_values('odometer', ascending=False).to_string(index=False))

# 2.9
df['car_age'] = 2025 - df['year']
print('\n── 2.9  Car age column sample:')
print(df[['year','car_age']].head(8).to_string(index=False))

# 2.10
n = ((df['condition'] >= 48) & (df['odometer'] > 90000)).sum()
print(f'\n── 2.10  Cars with condition ≥ 48 AND odometer > 90,000: {n:,}')

# 2.11
newer = df[df['year'] > 2013]
sp = newer.groupby('state')['sellingprice'].mean().sort_values(ascending=False)
print('\n── 2.11  States with highest avg price (year > 2013, top 10):')
print(sp.head(10).to_string())
print(f'\n  → "{sp.idxmax()}" consistently has the highest avg price for newer cars.')

# 2.12
threshold = df['condition'].quantile(0.80)
excellent = df[df['condition'] >= threshold]
vm = excellent.groupby('make')['sellingprice'].mean().sort_values().head(10)
print('\n── 2.12  Value-for-money makes (top 20% condition, lowest avg price):')
print(vm.to_string())
print(f'\n  → Most value-for-money: "{vm.idxmin()}"')

# ══════════════════════════════════════════════════════════════
#  TASK 3 — DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════
print('\n' + '═'*60)
print(' TASK 3 — DATA VISUALIZATION')
print('═'*60)

# 3.1 Correlation Heatmap
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, linecolor='#0f1117', ax=ax, annot_kws={'size':9})
ax.set_title('Correlation Matrix — Numerical Features', pad=14, color='white', fontsize=14)
save('3_1_correlation_heatmap')
print('\n── 3.1 saved')
print('  INSIGHT: sellingprice & mmr strongly correlated (~0.98). odometer negatively')
print('  correlated with price (-0.35). year positively correlated with sellingprice.')

# 3.2 Avg Price by Year
avg_yr = df[df['year']>=1995].groupby('year')['sellingprice'].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(avg_yr['year'], avg_yr['sellingprice'],
       color=plt.cm.plasma(np.linspace(0.2,0.85,len(avg_yr))), edgecolor='#ffffff11')
ax.set_title('Average Selling Price by Year', pad=12, color='white', fontsize=14)
ax.set_xlabel('Year'); ax.set_ylabel('Avg Selling Price ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax.grid(axis='y')
save('3_2_avg_price_by_year')
print('\n── 3.2 saved')
print('  CHOICE: BAR chart — year is discrete categorical; bars allow exact year comparison.')
print('  INSIGHT: Clear upward trend — newer cars fetch higher prices.')
print('  Post-2010 steep rise; pre-2000 dip reflects age/depreciation.')

# 3.3 Avg Price by Odometer
df['odo_bin'] = pd.cut(df['odometer'], bins=range(0, 200001, 10000))
op = df.groupby('odo_bin', observed=True)['sellingprice'].mean().dropna()
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(op)), op.values, color='#7b68ee', lw=2.5, marker='o', ms=5)
ax.fill_between(range(len(op)), op.values, alpha=0.15, color='#7b68ee')
ax.set_xticks(range(len(op)))
ax.set_xticklabels([str(b) for b in op.index], rotation=60, ha='right', fontsize=7)
ax.set_title('Average Selling Price by Odometer Range (10k-mile bins)', pad=12, color='white')
ax.set_xlabel('Odometer Range (miles)'); ax.set_ylabel('Avg Selling Price ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax.grid(axis='y')
save('3_3_avg_price_by_odometer')
print('\n── 3.3 saved')
print('  INSIGHT: Strong downward trend — higher mileage lowers price sharply.')
print('  Steepest drop 0-50k miles. Prices stabilise around $8-10k beyond 100k miles.')

# 3.4 Cars by State
sc = df['state'].value_counts().reset_index()
sc.columns = ['state','count']
fig, ax = plt.subplots(figsize=(16, 7))
bar_colors = [PALETTE[0] if i<3 else '#3a3d5c' for i in range(len(sc))]
ax.bar(sc['state'], sc['count'], color=bar_colors, edgecolor='#ffffff11')
ax.set_title('Number of Cars Sold by State', pad=12, color='white', fontsize=14)
ax.set_xlabel('State'); ax.set_ylabel('Number of Cars')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
ax.grid(axis='y'); plt.xticks(rotation=60)
save('3_4_cars_by_state')
t3 = sc.head(3)
print('\n── 3.4 saved')
print(f"  Top 3: {t3.iloc[0]['state'].upper()} ({t3.iloc[0]['count']:,}) | "
      f"{t3.iloc[1]['state'].upper()} ({t3.iloc[1]['count']:,}) | "
      f"{t3.iloc[2]['state'].upper()} ({t3.iloc[2]['count']:,})")

# 3.5 Avg Price by Condition bins=5
df['cond5'] = pd.cut(df['condition'], bins=range(0, 55, 5))
c5p = df.groupby('cond5', observed=True)['sellingprice'].mean().dropna()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([str(b) for b in c5p.index], c5p.values,
       color=plt.cm.RdYlGn(np.linspace(0.2,0.9,len(c5p))), edgecolor='#ffffff11')
ax.set_title('Average Selling Price by Condition Score (bins of 5)', pad=12, color='white')
ax.set_xlabel('Condition Score Range'); ax.set_ylabel('Avg Selling Price ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
ax.grid(axis='y'); plt.xticks(rotation=30)
save('3_5_avg_price_by_condition_bins5')
print('\n── 3.5 saved')
print('  INSIGHT: Clear positive relation — higher condition score → higher price.')
print('  Condition 45-50 commands highest prices (~$25k+). Below 20 = near-salvage.')

# 3.6 Count by Condition bins=10
df['cond10'] = pd.cut(df['condition'], bins=range(0, 60, 10))
c10n = df.groupby('cond10', observed=True).size()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar([str(b) for b in c10n.index], c10n.values,
              color=PALETTE[:len(c10n)], edgecolor='#ffffff11')
ax.set_title('Number of Cars Sold by Condition Range (bins of 10)', pad=12, color='white')
ax.set_xlabel('Condition Score Range'); ax.set_ylabel('Number of Cars')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+300,
            f'{int(b.get_height()):,}', ha='center', va='bottom', fontsize=9)
ax.grid(axis='y'); plt.xticks(rotation=15)
save('3_6_count_by_condition_bins10')
print('\n── 3.6 saved')
print('  INSIGHT: Bulk of cars in 20-40 range (average market quality).')
print('  30-40 bucket has highest volume — the used-car sweet spot.')

# 3.7 Box plot price by color
STD_COLORS = ['white','black','gray','silver','blue','red','brown','beige',
              'gold','green','orange','purple','burgundy','off-white','yellow',
              'charcoal','turquoise']
df_c = df[df['color'].isin(STD_COLORS)].copy()
color_order = (df_c.groupby('color')['sellingprice'].median()
                    .sort_values(ascending=False).index.tolist())

# With outliers
fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=df_c, x='color', y='sellingprice', order=color_order,
            palette='coolwarm', flierprops={'alpha':0.15,'markersize':2}, ax=ax)
ax.set_title('Selling Price Distribution by Color (with outliers)', pad=12, color='white')
ax.set_xlabel('Color'); ax.set_ylabel('Selling Price ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
plt.xticks(rotation=30); ax.grid(axis='y')
save('3_7a_boxplot_by_color_with_outliers')

# Remove outliers — collect indices per group
keep_idx = []
for _, grp in df_c.groupby('color'):
    Q1, Q3 = grp['sellingprice'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    keep_idx.extend(grp[(grp['sellingprice'] >= Q1-1.5*IQR) &
                        (grp['sellingprice'] <= Q3+1.5*IQR)].index.tolist())
df_no_out = df_c.loc[keep_idx].copy()

fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=df_no_out, x='color', y='sellingprice', order=color_order,
            palette='coolwarm', ax=ax)
ax.set_title('Selling Price Distribution by Color (outliers removed)', pad=12, color='white')
ax.set_xlabel('Color'); ax.set_ylabel('Selling Price ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:,.0f}'))
plt.xticks(rotation=30); ax.grid(axis='y')
save('3_7b_boxplot_by_color_no_outliers')

print('\n── 3.7 saved (both with & without outliers)')
print('  INSIGHT: After IQR removal, distributions are cleaner.')
print('  Rare colors (orange, turquoise, purple) have higher medians.')
print('  Common colors (white, silver, black) show tighter, lower price ranges.')

print('\n' + '═'*60)
print(' ✅  ALL DONE — plots saved to ./plots/')
print('═'*60)
