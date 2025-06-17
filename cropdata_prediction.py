
# --- 1. IMPORTS ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 2. LOAD & CLEAN DATA ---
df = pd.read_csv("/home/smirithi/Downloads/crop_dataset.csv")  # <- Update this to your actual file

# Drop nulls or handle missing
df.dropna(inplace=True)

# Pivot to align Area, Yield, and Production
pivot_df = df.pivot_table(index=["Area", "Item", "Year"], 
                          columns="Element", 
                          values="Value").dropna().reset_index()

# --- 3. FEATURE ENGINEERING ---
pivot_df['Productivity'] = pivot_df['Production'] / pivot_df['Area harvested']

# Label encoding
pivot_df['Area_Code'] = pivot_df['Area'].astype('category').cat.codes
pivot_df['Item_Code'] = pivot_df['Item'].astype('category').cat.codes

# Final feature set
X = pivot_df[['Area_Code', 'Item_Code', 'Year', 'Area harvested', 'Yield']]
y = pivot_df['Production']

# --- 4. EDA (Optional Visualizations) ---
sns.heatmap(pivot_df[['Area harvested', 'Yield', 'Production']].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Productivity comparison
top_items = pivot_df.groupby('Item')['Productivity'].mean().sort_values(ascending=False).head(10)
top_items.plot(kind='bar', title="Top 10 Productive Crops")
plt.ylabel("Production per Area")
plt.savefig("productivity_comparison.png")

# --- 5. MODEL TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "production_model.pkl")
pivot_df.to_csv("cleaned_data.csv", index=False)
