import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filepath = "data/robotex5.csv"
df = pd.read_csv(filepath)

# Convert 'start_time' to datetime format
df["start_time"] = pd.to_datetime(df["start_time"])
# sort by 'start_time'
df = df.sort_values("start_time")

# Extract features from 'start_time'
df["hour"] = df["start_time"].dt.hour
df["day_of_week"] = df["start_time"].dt.dayofweek  # Monday=0, Sunday=6
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Use KMeans to cluster 'start_lat' and 'start_lng' into spatial zones
kmeans = KMeans(n_clusters=5, random_state=42)  # Assuming 5 clusters for simplicity
df["zone"] = kmeans.fit_predict(df[["start_lat", "start_lng"]])

# Quick visualization of spatial clusters
plt.scatter(df["start_lng"], df["start_lat"], c=df["zone"], cmap="viridis")
plt.title("Spatial Clustering of Ride Demand Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
plt.savefig("spatial_clusters.png")
