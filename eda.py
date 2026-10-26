import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("keystroke_attempts_valid.csv")
print(df.info())
print(df.describe())
print(
    df.groupby("user_id")[["dwell_time_ms", "flight_time_ms"]]
      .median()
)

df["dwell_time_ms"].hist(bins=30)
plt.xlabel("Dwell time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Dwell Times")
# plt.show()

df["flight_time_ms"].dropna().hist(bins=50)
plt.xlabel("Flight time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Flight Times")
# plt.show()

# Filter inputs with flight times over 5 seconds
slow_inputs = df[df["flight_time_ms"] > 10000]
# Display relevant columns
print(slow_inputs[
    ["user_id", "attempt_id", "key", "press_time",
     "release_time", "flight_time_ms"]
].sort_values("flight_time_ms", ascending=False))

print(
    slow_inputs.groupby(["user_id", "attempt_id"]).size()
)