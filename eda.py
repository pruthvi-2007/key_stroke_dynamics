import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("keystroke_attempts_valid.csv")
print(df.info())
print(df.describe())
print(
    df.groupby("user_id")[["dwell_time_ms", "flight_time_ms"]]
      .median()
)

import matplotlib.pyplot as plt

df["dwell_time_ms"].hist(bins=30)
plt.xlabel("Dwell time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Dwell Times")
plt.show()

df["flight_time_ms"].dropna().hist(bins=50)
plt.xlabel("Flight time (ms)")
plt.ylabel("Frequency")
plt.title("Distribution of Flight Times")
plt.show()