import pandas as pd
df = pd.read_csv("keystroke_attempts_fixed.csv")
#preserve og
raw_df = df.copy()
#print(df.head())

#sort the dataset on the basis of user id, attempt id
df = df.sort_values(by = ["user_id", "attempt_id"])
#print(df.head())

#check for duplicates and delete if they exist
print(df.duplicated().sum())

#count number of attempts per user
print(df.value_counts("user_id")/43)
print(df.groupby("user_id")["attempt_id"].nunique())

sizes = df.groupby(["user_id", "attempt_id"]).size()
print(sizes[sizes >43])
#find the attempts with more than 43 rows
attempt_sizes = df.groupby(
    ["user_id", "attempt_id"]
).size()
#print(attempt_sizes.value_counts().sort_index())


#find missing attempts for each user
expected = set(range(1, 201))
for user in df["user_id"].unique():
    existing = set(
        df.loc[df["user_id"] == user, "attempt_id"].unique()
    )

    missing = sorted(expected - existing)

    #print(user, "missing attempts:", missing)

#merge missing attempts of User2

#change wrong attempt numbers for user3
# Select User3's attempts numbered 201–300
#mask = (
#    (df["user_id"] == "User3") &
#    (df["attempt_id"].between(201, 300)))
# Subtract 100 from those attempt numbers
#df.loc[mask, "attempt_id"] -= 100

#get statistics for each user
#print(df.groupby("user_id")["key"].value_counts())

#check for missing values
#print(df.isna().sum())
#negative dwell times and flight times. press time>release time
#identify rows with the missing values
#print(df[df["flight_time_ms"].isna()][
#    ["user_id", "attempt_id", "key", "flight_time_ms"]])



#save changes in new file
df.to_csv("keystroke_attempts_fixed.csv", index=False)