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

category_0 = df_pca[df_to_classify["maintenance_category"] == 0]
category_1 = df_pca[df_to_classify["maintenance_category"] == 1]

cond = df_to_classify[df_to_classify["maintenance_category"] == 1].index
df_bad = df_pca[cond.tolist()]

rightmost_point_index = np.argmax(df_bad[:, 0])
rightmost_point = df_bad[rightmost_point_index]
print(rightmost_point)

plt.figure(figsize=(10, 6))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_to_classify['maintenance_category'], edgecolor='k', s=50)
plt.scatter(
    category_0[:, 0],
    category_0[:, 1],
    color="blue",
    label="High Maintenance",
    edgecolor="k",
    s=50,
)

plt.scatter(
    category_1[:, 0],
    category_1[:, 1],
    color="yellow",
    label="Low Maintenance",
    edgecolor="k",
    s=50,
)

plt.scatter(
    rightmost_point[0],
    rightmost_point[1],
    color="red",
    edgecolor="k",
    s=100,
    label="Rightmost Point From Yellow Cluster",
)
plt.axvline(
    x=rightmost_point[0],
    color="red",
    linestyle="--",
    label="Line of separation",
    linewidth=2,
)


# Highlight the rightmost point
# plt.scatter(df_pca[1804,0], df_pca[1804,1], color='blue', edgecolor='k', s=150, label="mindspore-ai/models")
# print(" outlier ",df_to_classify.iloc[1804])


plt.title("2D PCA Projection with K-Means Cluster Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.legend()
plt.show()


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
