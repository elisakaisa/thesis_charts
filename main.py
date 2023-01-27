import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# import data
df = pd.read_csv("Recording.txt", sep= ",")

#delete first 34 rows & last 30
df.drop(index=df.index[:34],inplace=True)
df.drop(index=df.tail(30).index,inplace=True)

#rename columns
df.rename(columns={"left wrist x": "left wrist x raw", "left wrist y": "left wrist y raw"}, inplace=True)

# invert y coordinate
df["left wrist y raw"] = df["left wrist y raw"].apply(lambda y: 400 - y)

#change timestamp
t_zero = df["time"].iloc[0]
df["time"] = df["time"].apply(lambda t: (t - t_zero) / 1000)

# add moving average columns
df["left wrist y (3)"] = df["left wrist y raw"].rolling(3).mean()
df["left wrist x (3)"] = df["left wrist x raw"].rolling(3).mean()
df["left wrist y (10)"] = df["left wrist y raw"].rolling(10).mean()
df["left wrist x (10)"] = df["left wrist x raw"].rolling(10).mean()
df["left wrist y (30)"] = df["left wrist y raw"].rolling(30).mean()
df["left wrist x (30)"] = df["left wrist x raw"].rolling(30).mean()


# Savitzky-golay filter
x = df["left wrist x raw"].to_numpy()
y = df["left wrist y raw"].to_numpy()
x_smooth = signal.savgol_filter(x, window_length=4, polyorder=3, mode="nearest")
y_smooth = signal.savgol_filter(y, window_length=4, polyorder=3, mode="nearest")
df["left wrist x (SG w4 p3)"] = pd.DataFrame(x_smooth)
df["left wrist y (SG w4 p3)"] = pd.DataFrame(y_smooth)
df["left wrist x (SG w4 p3)"] = df["left wrist x (SG w4 p3)"].shift(34)
df["left wrist y (SG w4 p3)"] = df["left wrist y (SG w4 p3)"].shift(34)

x = df["left wrist x raw"].to_numpy()
y = df["left wrist y raw"].to_numpy()
x_smooth = signal.savgol_filter(x, window_length=10, polyorder=4, mode="nearest")
y_smooth = signal.savgol_filter(y, window_length=10, polyorder=4, mode="nearest")
df["left wrist x (SG w10 p4)"] = pd.DataFrame(x_smooth)
df["left wrist y (SG w10 p4)"] = pd.DataFrame(y_smooth)
df["left wrist x (SG w10 p4)"] = df["left wrist x (SG w10 p4)"].shift(34)
df["left wrist y (SG w10 p4)"] = df["left wrist y (SG w10 p4)"].shift(34)

# Plots for visualizing the data
fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axes = np.reshape(axes, -1)

axes[0].plot(df['left wrist x raw'], df['left wrist y raw'])
axes[0].set(title="Left wrist movement, raw data", xlabel="x",ylabel="y")
axes[0].legend()

axes[1].plot(df['left wrist x (3)'], df['left wrist y (3)'])
axes[1].set(title="Left wrist movement, moving avg, window 3", xlabel="x",ylabel="y")
axes[1].legend()

axes[2].plot(df['left wrist x (10)'], df['left wrist y (10)'])
axes[2].set(title="Left wrist movement, moving avg, window 10", xlabel="x",ylabel="y")
axes[2].legend()

axes[3].plot(df['left wrist x (30)'], df['left wrist y (30)'])
axes[3].set(title="Left wrist movement, moving avg, window 30", xlabel="x",ylabel="y")
axes[3].legend()

axes[4].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'])
axes[4].set(title="Left wrist movement, SA, window 4, polyorder 3", xlabel="x",ylabel="y")
axes[4].legend()

axes[5].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'])
axes[5].set(title="Left wrist movement, SA, window 10, polyorder 4", xlabel="x",ylabel="y")
axes[5].legend()


fig2, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
axes = np.reshape(axes, -1)

axes[0].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw data")
axes[0].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA, window of 3")
axes[0].plot(df['left wrist x (10)'], df['left wrist y (10)'], label="MA, window of 10")
axes[0].plot(df['left wrist x (30)'], df['left wrist y (30)'], label="MA, window of 30")
axes[0].set(title="Left wrist movement", xlabel="x",ylabel="y")
axes[0].legend()

axes[1].plot(df['time'], df['left wrist x raw'], label="raw data")
axes[1].plot(df['time'], df['left wrist x (3)'], label="MA, window of 3")
axes[1].plot(df['time'], df['left wrist x (10)'], label="MA, window of 10")
axes[1].plot(df['time'], df['left wrist x (30)'], label="MA, window of 30")
axes[1].set(title="Left wrist movement", xlabel="time [s]",ylabel="x coordinate")
axes[1].legend()

axes[2].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw")
axes[2].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'], label="SG (window 4, polyorder 3")
axes[2].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'], label="SG (window 10, polyorder 4")
axes[2].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA (window 3")
axes[2].set(title="Left wrist movement, SG, window 4, polyorder 3", xlabel="x",ylabel="y")
axes[2].legend()

axes[3].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw data")
axes[3].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA, window of 3")
axes[3].plot(df['left wrist x (10)'], df['left wrist y (10)'], label="MA, window of 10")
axes[3].plot(df['left wrist x (30)'], df['left wrist y (30)'], label="MA, window of 30")
axes[3].set(title="Left wrist movement", xlabel="x",ylabel="y")
axes[3].set_ylim(300, 400)
axes[3].set_xlim(250, 360)
axes[3].legend()

axes[4].plot(df['time'], df['left wrist x raw'], label="raw data")
axes[4].plot(df['time'], df['left wrist x (3)'], label="MA, window of 3")
axes[4].plot(df['time'], df['left wrist x (10)'], label="MA, window of 10")
axes[4].plot(df['time'], df['left wrist x (SG w4 p3)'], label="SG, window 4, polyorder 3")
axes[4].plot(df['time'], df['left wrist x (SG w10 p4)'], label="SG, window 10, polyorder 4")
axes[4].set(title="Left wrist movement", xlabel="time [s]",ylabel="x coordinate")
axes[4].legend()

axes[5].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw")
axes[5].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'], label="SG (window 4, polyorder 3")
axes[5].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'], label="SG (window 10, polyorder 4")
axes[5].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA (window 3)")
axes[5].set(title="Left wrist movement, SG, window 4, polyorder 3", xlabel="x",ylabel="y")
axes[5].set_ylim(300, 400)
axes[5].set_xlim(250, 360)
axes[5].legend()



plt.show()

print(df)