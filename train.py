import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


filepath = "data/robotex5_clustered.csv"
df = pd.read_csv(filepath)

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


# TODO: use same colors
# Plot 1: Most popular time of day per zone (number of rides by hour)
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=rides_by_hour_zone, x="hour", y="num_rides", hue="zone", palette="Set1"
)
plt.title("Most Popular Time of Day per Zone (Number of Rides)")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Rides")
plt.legend(title="Zone")
plt.grid(True)
plt.savefig("most_popular_time_of_day.png")

# Plot 2: Average ride value by hour and zone
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=rides_by_hour_zone, x="hour", y="avg_ride_value", hue="zone", palette="Set2"
)
plt.title("Average Ride Value by Hour and Zone")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Ride Value (currency)")
plt.legend(title="Zone")
plt.grid(True)
plt.savefig("average_ride_value.png")


# # Prepare the feature matrix and target vector
# # 5 zones x 24 hours x 7 days = 840 data points
# X = aggregated_data[["zone", "hour", "day_of_week"]]
# # TODO: normalize the data?
# y = aggregated_data["num_rides"]

# X_encoded = pd.get_dummies(X, columns=["zone"])
# # TODO: get dummies for hours and days of the week?

# # TODO: should split the data before aggregating, otherwise we loose the information about some of the zones
# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_encoded, y, test_size=0.3, random_state=42
# )

# # Initialize and fit the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Making predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"R^2 Score: {r2:.2f}")

# # Save the model
# import joblib

# joblib.dump(model, "model.pkl")
# print("Model trained and saved.")

# # TODO: find a way to visualize  model's performance
# # # plot
# # plt.scatter(y_test, y_pred)
# # plt.xlabel("True Values")
# # plt.ylabel("Predictions")
# # plt.title("True Values vs. Predictions")
# # plt.show()

# # TODO: there is also ride_value column in the data, maybe we can use it to predict the average ride value