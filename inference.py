import pandas as pd
import numpy as np

# Load aggregated data from the historical dataset (already calculated)
# This data contains aggregated number of rides and average ride values for each zone, hour, and day
aggregated_data = pd.read_csv("data/aggregated_demand.csv")

# Load the cluster centers with colors
cluster_centers = pd.read_csv("data/cluster_centers.csv")

# Driver's current location (for example)
driver_lat, driver_lng = 59.43, 24.75

# Get current time details (simulating current hour and day of the week)
current_hour = 14
current_day_of_week = 3  # For example, Wednesday

# Filter the historical data for the current hour and day
current_data = aggregated_data[
    (aggregated_data["hour"] == current_hour)
    & (aggregated_data["day_of_week"] == current_day_of_week)
]

# Merge current_data with cluster_centers to get zone locations
current_data = current_data.merge(cluster_centers, on="zone")

# Calculate the distance from the driver to each zone center
current_data["distance"] = np.sqrt(
    (current_data["center_lat"] - driver_lat) ** 2
    + (current_data["center_lng"] - driver_lng) ** 2
)

# Sort zones by demand (num_rides), ride value (avg_ride_value), and distance
# You can use a weighted sum or scoring system based on your preference
current_data["score"] = (
    current_data["num_rides"]
    * current_data["avg_ride_value"]
    / current_data["distance"]
)
sorted_zones = current_data.sort_values(by="score", ascending=False)

# Show the top recommendations
print(sorted_zones[["zone", "num_rides", "avg_ride_value", "distance", "score"]].head())

# Save the sorted zones to a file if needed
sorted_zones.to_csv("recommended_zones.csv", index=False)
