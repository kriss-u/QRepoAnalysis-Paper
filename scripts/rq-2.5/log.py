import pandas as pd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

json_file_path = "commit_analysis.json"
data = pd.read_json(json_file_path)
df_to_classify = data.copy()


features = [
    "commit_frequency",
    "avg_days_between_commits",
    "num_commits",
    "num_unique_authors",
    "num_months",
]
print(f"df_to_classify : {df_to_classify[features]}")
print()

for fe in features:
    df_to_classify[fe + "_extra"] = df_to_classify[fe]

for fe in features:
    df_to_classify[fe] = np.log(df_to_classify[fe] + 1)
print(f"df_to_classify : {df_to_classify[features]}")
print()

df_cluster = df_to_classify[features]
df_cluster = df_cluster[features].dropna()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)
print(f"df_scaled : ", df_scaled)
print()

kmeans = KMeans(n_clusters=2, random_state=40)
df_to_classify = df_to_classify.dropna(subset=features)
df_to_classify["maintenance_category"] = kmeans.fit_predict(df_scaled)


"""2d"""

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(
    df_pca[:, 0],
    df_pca[:, 1],
    c=df_to_classify["maintenance_category"],
    edgecolor="k",
    s=50,
)
plt.colorbar()
plt.title("KMeans Clustering (PCA Reduced to 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()


# # Add the PCA result back to the DataFrame for reference
# df_to_classify['PC1'] = df_pca[:, 0]
# df_to_classify['PC2'] = df_pca[:, 1]
# # Create an interactive scatter plot using Plotly
# fig = px.scatter(df_to_classify, x='PC1', y='PC2', text=df_to_classify.index)
# # Add hover information
# fig.update_traces(textposition='top center', hovertemplate="Index: %{text}<br>PC1: %{x}<br>PC2: %{y}<extra></extra>")
# fig.show()


"""stats"""

df_to_classify["maintenance_category"] = df_to_classify["maintenance_category"].map(
    {0: "High maintenance", 1: "Low maintenance"}
)
cluster_size = df_to_classify["maintenance_category"].value_counts().sort_index()
print(cluster_size, cluster_size / len(df_to_classify))
print()

centroids = df_to_classify.groupby("maintenance_category")[features].mean()
print(centroids)
print()

"""centroid on the original"""
features_extra = [
    "commit_frequency_extra",
    "avg_days_between_commits_extra",
    "num_commits_extra",
    "num_unique_authors_extra",
    "num_months_extra",
]
centroids = df_to_classify.groupby("maintenance_category")[features_extra].mean()
print(centroids)
print()


"""print hovering"""
for fe in df_to_classify.keys():
    print(f"{fe} : {df_to_classify.iloc[5199][fe]}")


# columns = ["repo_name", "commit_frequency_extra", "avg_days_between_commits_extra", "num_commits_extra", "num_unique_authors_extra",
#            "maintenance_category", "num_months"]

# # Extracting the specific rows and columns
# extracted_df = df_to_classify.loc[[1804,115,22,5199,26,1846,94,46,2153,2740,3174,2482], columns]
# extracted_df.to_csv("anomalies.csv",index=False)
