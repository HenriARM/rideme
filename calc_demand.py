import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# Load the ride data with zones
df = pd.read_csv("data/robotex5_clustered.csv")

# Load the cluster centers with colors
cluster_centers = pd.read_csv("data/cluster_centers.csv")

# Create a color map from the cluster_centers file
zone_colors = cluster_centers.set_index("zone")["color"].to_dict()


# Extract features from 'start_time'
df["start_time"] = pd.to_datetime(df["start_time"])
df = df.sort_values("start_time")
df["hour"] = df["start_time"].dt.hour
df["day_of_week"] = df["start_time"].dt.dayofweek  # Monday=0, Sunday=6
# df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# Aggregating the demand data (number of rides) and average ride value for each cluster and hour of the day
aggregated_data = (
    df.groupby(["zone", "hour", "day_of_week"])
    .agg(
        num_rides=("ride_value", "count"),  # Count of rides as a measure of demand
        avg_ride_value=("ride_value", "mean"),  # Average ride value in each group
    )
    .reset_index()
)

# Aggregating number of rides by zone and hour
rides_by_hour_zone = (
    df.groupby(["zone", "hour"])
    .agg(num_rides=("ride_value", "count"), avg_ride_value=("ride_value", "mean"))
    .reset_index()
)


# Plot 1: Most popular time of day per zone (number of rides by hour)
plt.figure(figsize=(12, 6))
for zone in rides_by_hour_zone["zone"].unique():
    zone_data = rides_by_hour_zone[rides_by_hour_zone["zone"] == zone]
    plt.plot(
        zone_data["hour"],
        zone_data["num_rides"],
        label=f"Zone {zone}",
        color=zone_colors[zone],
    )
plt.title("Most Popular Time of Day per Zone (Number of Rides)")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Rides")
plt.legend(title="Zone")
plt.grid(True)
plt.savefig("most_popular_time_of_day.png")

# Plot 2: Average ride value by hour and zone
plt.figure(figsize=(12, 6))
for zone in rides_by_hour_zone["zone"].unique():
    zone_data = rides_by_hour_zone[rides_by_hour_zone["zone"] == zone]
    plt.plot(
        zone_data["hour"],
        zone_data["avg_ride_value"],
        label=f"Zone {zone}",
        color=zone_colors[zone],
    )
plt.title("Average Ride Value by Hour and Zone")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Ride Value (currency)")
plt.legend(title="Zone")
plt.grid(True)
plt.savefig("average_ride_value.png")