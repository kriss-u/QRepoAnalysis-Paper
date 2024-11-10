import os
import pandas as pd
import networkx as nx
from itertools import combinations
import community as community_louvain
import matplotlib.pyplot as plt
from collections import Counter

repo_names = [
    "Qiskit/qiskit",
    "Qiskit/qiskit-aer",
    "quantumlib/Cirq",
    "qutip/qutip",
    "amazon-braket/amazon-braket-sdk",
    "PennyLaneAI/pennylane",
    "microsoft/qsharp",
    "quantumlib/openfermion",
    "rigetticomputing/pyquil",
    "openqasm/openqasm",
    "Qiskit/qiskit-aqua",
]

G = nx.Graph()
root_dir = "./"

contributor_repos = {}

for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".parquet"):
            file_path = os.path.join(subdir, file)
            df = pd.read_parquet(file_path)

            for index, row in df.iterrows():
                if row["repo_id"] not in repo_names:
                    continue

                contributors = row["contributors"]
                if contributors is not None:
                    for contributor in contributors:
                        if contributor not in G:
                            G.add_node(contributor)
                        if contributor not in contributor_repos:
                            contributor_repos[contributor] = []
                        contributor_repos[contributor].append(row["repo_id"])

                    # Create edges between pairs of contributors for each repository
                    for contributor1, contributor2 in combinations(contributors, 2):
                        if not G.has_edge(contributor1, contributor2):
                            G.add_edge(contributor1, contributor2, weight=1)
                        else:
                            G[contributor1][contributor2]["weight"] += 1


# Calculate number of frameworks used by each contributor
framework_count = {
    contributor: len(set(repos)) for contributor, repos in contributor_repos.items()
}
# Find the top 5 contributors with the highest number of frameworks
top_5_contributors = sorted(framework_count.items(), key=lambda x: x[1], reverse=True)[
    :5
]
# Assign unique shapes to the top 5 contributors
node_shapes = ["o", "s", "^", "D", "p"]  # Circle, square, triangle, diamond, pentagon
top_5_contributors_dict = {
    contributor: node_shapes[i] for i, (contributor, _) in enumerate(top_5_contributors)
}


# Louvain community detection
partition = community_louvain.best_partition(G, weight="weight", resolution=1)
unique_communities = list(set(partition.values()))
colors = plt.cm.get_cmap("tab20", len(unique_communities))
community_color_map = {
    community_id: colors(i) for i, community_id in enumerate(unique_communities)
}

degree_centrality = nx.degree_centrality(G)

external_cluster_connections = {}

for node in G.nodes:
    node_cluster = partition[node]
    unique_external_clusters = set()

    for neighbor in G.neighbors(node):
        neighbor_cluster = partition[neighbor]

        if neighbor_cluster != node_cluster:
            unique_external_clusters.add(neighbor_cluster)

    external_cluster_connections[node] = len(unique_external_clusters)

# Filter and store nodes with >= 2 external clusters in a list of tuples
significant_external_nodes = [
    (
        node,
        framework_count[node],
        G.degree[node],
        degree_centrality[node],
        partition[node],
    )
    for node in G.nodes
    if framework_count[node] >= 2
]

# Sort by external clusters (ascending) and centrality (ascending) if clusters are the same
significant_external_nodes.sort(key=lambda x: (x[1], degree_centrality[x[0]]))

print("Nodes with 2 or More Frameworks, Sorted by Framework Count and Centrality:")

dfs = {}
dfs["node"] = []
dfs["num_frameworks"] = []
dfs["degree"] = []
dfs["centrality"] = []
dfs["cluster_id"] = []


for node, num_frameworks, degree, centrality, cluster_id in significant_external_nodes:
    print(
        f"Node: {node}, Framework count: {num_frameworks}, Degree: {degree}, Centrality: {centrality:.4f}, Cluster ID: {cluster_id}"
    )
    dfs["node"].append(node)
    dfs["num_frameworks"].append(num_frameworks)
    dfs["degree"].append(degree)
    dfs["centrality"].append(centrality)
    dfs["cluster_id"].append(cluster_id)

dfs = pd.DataFrame(dfs)
dfs.to_csv("top.csv", index=False)

# Calculate positions for nodes using spring layout
pos = nx.spring_layout(G, k=0.15, iterations=20)

node_sizes = [100 for node in G.nodes]
# node_sizes = []
# for node in G.nodes:
#     if framework_count[node] == 0:
#         node_sizes.append(5 * framework_count[node])
#     elif framework_count[node] == 1:
#         node_sizes.append(15 * framework_count[node])
#     elif framework_count[node] == 2:
#         node_sizes.append(35 * framework_count[node])
#     elif framework_count[node] == 3:
#         node_sizes.append(55 * framework_count[node])
#     elif framework_count[node] >= 4:
#         node_sizes.append(100 * framework_count[node])


plt.figure(figsize=(12, 12))

for community_id in unique_communities:
    community_nodes = [
        node for node, comm_id in partition.items() if comm_id == community_id
    ]

    community_nodes = [
        node
        for node, comm_id in partition.items()
        if comm_id == community_id
        and not node
        in ["dependabot[bot]", "eendebakpt", "JiahaoYao", "ryanhill1", "nathanshammah"]
    ]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=community_nodes,
        node_size=[
            node_sizes[i]
            for i in range(len(G.nodes))
            if list(G.nodes())[i] in community_nodes
        ],
        node_color=[community_color_map[community_id]] * len(community_nodes),
        alpha=0.8,
    )

nx.draw_networkx_edges(G, pos, width=0.05, alpha=0.02, edge_color="gray")

handles = []
labels = []
for i, (contributor, shape) in enumerate(top_5_contributors_dict.items()):
    if contributor in G.nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[contributor],
            node_size=400,
            node_color=community_color_map[partition[contributor]],
            node_shape=shape,
            label=f"{contributor} ({framework_count[contributor]})",
        )
        if f"{contributor} ({framework_count[contributor]})" not in labels:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=shape,
                    color="w",
                    label=f"{contributor}, #frameworks : {framework_count[contributor]}",
                    markerfacecolor=community_color_map[partition[contributor]],
                    markersize=10,
                )
            )
            labels.append(
                f"{contributor}, #frameworks : {framework_count[contributor]}"
            )

plt.legend(handles=handles, loc="lower left", fontsize=16)


community_repos = {}
for author, community_id in partition.items():
    repos = contributor_repos.get(author, [])
    if repos:
        if community_id not in community_repos:
            community_repos[community_id] = []
        community_repos[community_id].extend(repos)

top_repos_per_community = {}
for community_id, repos in community_repos.items():
    repo_counts = Counter(repos)
    top_repos = repo_counts.most_common(3)
    top_repos_per_community[community_id] = top_repos

for community_id, top_repos in top_repos_per_community.items():
    community_nodes = [
        node for node, comm_id in partition.items() if comm_id == community_id
    ]
    community_positions = [pos[node] for node in community_nodes]
    if community_positions:
        avg_x = sum([p[0] for p in community_positions]) / len(community_positions)
        avg_y = sum([p[1] for p in community_positions]) / len(community_positions)

        repo_names_display = [f"{repo} ({count})" for repo, count in top_repos]
        num_nodes = len(community_nodes)
        repo_label = (
            f"Community {community_id}:\n"
            + "\n".join(repo_names_display)
            + f"\nNodes: {num_nodes}"
        )

        plt.text(
            avg_x,
            avg_y,
            repo_label,
            fontsize=18,
            ha="center",
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.5"),
        )

plt.title("Contributor Collaboration Network with Community Information", fontsize=20)
plt.axis("off")

plt.show()

author_community_data = [
    {"author": author, "community_id": community_id}
    for author, community_id in partition.items()
]
df_author_community = pd.DataFrame(author_community_data)
df_author_community.to_csv("author_community_associations.csv", index=False)
nx.write_graphml(G, "contributor_collaboration_graph.graphml")


print("Top 5 Contributors with Framework Count and Cluster ID:")
for contributor, _ in top_5_contributors:
    cluster_id = partition.get(contributor, "N/A")
    num_frameworks = framework_count.get(contributor, 0)
    print(
        f"Contributor: {contributor}, Framework Count: {num_frameworks}, Cluster ID: {cluster_id}"
    )
