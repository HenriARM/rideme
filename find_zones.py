import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filepath = "data/robotex5.csv"
df = pd.read_csv(filepath)

# Convert 'start_time' to datetime format
df["start_time"] = pd.to_datetime(df["start_time"])
# sort by 'start_time'
df = df.sort_values("start_time")

# Use KMeans to cluster 'start_lat' and 'start_lng' into spatial zones
kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 clusters for simplicity
df["zone"] = kmeans.fit_predict(df[["start_lat", "start_lng"]])
df.to_csv("robotex5_clustered.csv", index=False)

# Save the clusters and centers to CSV for future use
cluster_centers = kmeans.cluster_centers_
cluster_data = pd.DataFrame(cluster_centers, columns=["center_lat", "center_lng"])
cluster_data["zone"] = cluster_data.index
cluster_data.to_csv("cluster_centers.csv", index=False)

# Quick visualization of spatial clusters
plt.scatter(df["start_lng"], df["start_lat"], c=df["zone"], cmap="viridis")
plt.scatter(
    cluster_data["center_lng"],
    cluster_data["center_lat"],
    c="red",
    marker="x",
    label="Cluster Centers",
)
plt.title("Spatial Clustering of Ride Demand Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.savefig("spatial_clusters.png")
