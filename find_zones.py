import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
filepath = "data/robotex5.csv"
df = pd.read_csv(filepath)

# Convert 'start_time' to datetime format
df["start_time"] = pd.to_datetime(df["start_time"])

# Sort by 'start_time'
df = df.sort_values("start_time")

# Use KMeans to cluster 'start_lat' and 'start_lng' into spatial zones
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # 5 clusters for simplicity
df["zone"] = kmeans.fit_predict(df[["start_lat", "start_lng"]])

# Save the clustered data
df.to_csv("robotex5_clustered.csv", index=False)

# Calculate cluster centers
cluster_centers = kmeans.cluster_centers_

# Assign colors to each zone (you can change colors as needed)
colors = sns.color_palette(
    "Set1", n_clusters
).as_hex()  # 'Set1' provides distinguishable colors
color_map = {i: colors[i] for i in range(n_clusters)}  # Map zone index to color

# Save the clusters and centers to CSV, including color information
cluster_data = pd.DataFrame(cluster_centers, columns=["center_lat", "center_lng"])
cluster_data["zone"] = cluster_data.index
cluster_data["color"] = cluster_data["zone"].map(color_map)  # Add the color mapping

cluster_data.to_csv("cluster_centers.csv", index=False)

# Quick visualization of spatial clusters using consistent colors
plt.figure(figsize=(10, 6))
for zone in range(n_clusters):
    zone_data = df[df["zone"] == zone]
    plt.scatter(
        zone_data["start_lng"],
        zone_data["start_lat"],
        color=color_map[zone],
        label=f"Zone {zone}",
    )

# Plot the cluster centers
plt.scatter(
    cluster_data["center_lng"],
    cluster_data["center_lat"],
    c="black",
    marker="x",
    s=100,
    label="Cluster Centers",
)
plt.title("Spatial Clustering of Ride Demand Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.savefig("spatial_clusters.png")