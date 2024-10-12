import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Load data
filepath = "data/robotex5.csv"
df = pd.read_csv(filepath)
df["start_time"] = pd.to_datetime(df["start_time"])
df = df.sort_values("start_time")

# Extract relevant features for clustering (latitude and longitude)
X = df[["start_lat", "start_lng"]]

# Range of possible cluster numbers (adjust as needed)
cluster_range = range(5, 8)

# Store inertia (sum of squared distances) and silhouette scores for each cluster number
inertia_values = []
silhouette_scores = []

# Define a sample size for silhouette score (use a smaller subset of the data)
sample_size = min(
    50000, len(X)
)  # Use a max of 50,000 points or the dataset size if smaller
X_sample = X.sample(n=sample_size, random_state=42)

# Set to True if you want to use MiniBatchKMeans for large datasets
use_mini_batch = False

# Loop over different cluster numbers
for n_clusters in cluster_range:
    # Perform KMeans or MiniBatchKMeans clustering
    if use_mini_batch:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=10000
        )
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    cluster_labels = kmeans.fit_predict(X)

    # Calculate inertia (sum of squared distances to the closest centroid)
    inertia_values.append(kmeans.inertia_)

    # Calculate silhouette score on the sampled data
    cluster_labels_sample = kmeans.predict(X_sample)  # Predict labels for the sample
    silhouette_avg = silhouette_score(X_sample, cluster_labels_sample)
    silhouette_scores.append(silhouette_avg)

    print(
        f"For {n_clusters} clusters: Inertia={kmeans.inertia_}, Silhouette={silhouette_avg}"
    )

# Plot the Elbow Method results (Inertia)
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, inertia_values, "bx-")
plt.xlabel("Number of Zones (Clusters)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method For Optimal Number of Zones")
plt.grid(True)
plt.show()

# Plot the Silhouette Score results
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, "bo-")
plt.xlabel("Number of Zones (Clusters)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal Number of Zones (Sampled Data)")
plt.grid(True)
plt.show()

# Optional: Save the results for further analysis
optimal_zone_data = pd.DataFrame(
    {
        "n_clusters": cluster_range,
        "inertia": inertia_values,
        "silhouette_score": silhouette_scores,
    }
)

optimal_zone_data.to_csv("optimal_zones_analysis.csv", index=False)

# Print the best number of zones according to the silhouette score
optimal_n_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print(f"Optimal number of zones (based on Silhouette Score): {optimal_n_clusters}")
