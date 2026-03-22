import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATH = Path("add path")
OUTPUT_DIR = Path("add path")

# data from csv into df 
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";").sort_values(["product", "timestamp"]).reset_index(drop=True)
    return df

# print(load_data(CSV_PATH))

# Sorts data into product specific dataframes (emeralds and tomatoes)
def sort_data(df: pd.DataFrame):
    df1 = df[df["product"] == "EMERALDS"].copy()
    df2 = df[df["product"] == "TOMATOES"].copy()
    return df1, df2

df = load_data(CSV_PATH)
df_emeralds, df_tomatoes = sort_data(df)

# print(df_emeralds)
# print(df_tomatoes)


def mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df_mid_price = df[["timestamp", "mid_price"]].copy()
    df_mid_price = df_mid_price.set_index("timestamp")
    return df_mid_price

df_tomatoes_mid_price = mid_price(df_tomatoes)
print(df_tomatoes_mid_price)

# This is the mid price over time with no smoothing, so it's very noisy, lots of ticks
# df_tomatoes_mid_price.plot()
# plt.show()

# this is the mid price with a rolling 80 tick window mean
def smooth_R_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df_smooth_mid_price = df.rolling(window=80).mean()
    return df_smooth_mid_price

def smooth_E_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    df_smooth_mid_price = df.ewm(span=80).mean()
    return df_smooth_mid_price


df_tomatoes_smooth_mid_price_r = smooth_R_mid_price(df_tomatoes_mid_price)
# df_tomatoes_smooth_mid_price_r.plot()
# plt.show()

df_tomatoes_smooth_mid_price_e = smooth_E_mid_price(df_tomatoes_mid_price)

# df_tomatoes_smooth_mid_price_e.plot()
plt.figure(figsize=(20, 15))
plt.plot(df_tomatoes_mid_price, label="Mid Price_raw")
plt.plot(df_tomatoes_smooth_mid_price_e, label="EWM")
plt.plot(df_tomatoes_smooth_mid_price_r, label="Rolling Mean")
plt.legend()
plt.title("Exponentially Weighted Moving Average (EWM) of Mid Price of Tomatoes")
plt.xlabel("Timestamp")
plt.ylabel("Mid Price")
plt.grid(True)
plt.show()



