import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# Load dataset
df = pd.read_csv("data/Cleaned_mall_customer_data.csv")

# KMeans clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Save KMeans model
with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
# ---- FIXED APRIORI PART ----
df['Income_Category'] = pd.cut(df['Annual Income (k$)'], bins=5, labels=False)
df['Spending_Category'] = pd.cut(df['Spending Score (1-100)'], bins=5, labels=False)

# One-hot encode safely
basket = pd.get_dummies(df[['Income_Category', 'Spending_Category']])

# Ensure binary/boolean values
basket = basket.apply(lambda col: col.map(lambda x: 1 if x > 0 else 0))
basket = basket.astype(bool)

# Run Apriori
frequent_items = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)

with open("models/apriori_rules.pkl", "wb") as f:
    pickle.dump(rules, f)




import pickle

import pandas as pd
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import pickle

# Load dataset
df = pd.read_csv("data/Cleaned_mall_customer_data.csv")

# KMeans clustering
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)



