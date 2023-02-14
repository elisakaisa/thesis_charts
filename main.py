import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
fig1, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 12))
#fig1.suptitle("Left wrist semi-circle movement filtered with different filters", fontsize=20)
axes = np.reshape(axes, -1)

axes[0].plot(df['left wrist x raw'], df['left wrist y raw'])
axes[0].set(title="a) raw data", xlabel="x",ylabel="y")
axes[0].set_aspect('equal', adjustable='datalim')
axes[0].legend()

axes[1].plot(df['left wrist x (3)'], df['left wrist y (3)'])
axes[1].set(title="b) moving avg, window 3", xlabel="x",ylabel="y")
axes[1].set_aspect('equal', adjustable='datalim')
axes[1].legend()

axes[2].plot(df['left wrist x (10)'], df['left wrist y (10)'])
axes[2].set(title="c) moving avg, window 10", xlabel="x",ylabel="y")
axes[2].set_aspect('equal', adjustable='datalim')
axes[2].legend()

axes[3].plot(df['left wrist x (30)'], df['left wrist y (30)'])
axes[3].set(title="d) moving avg, window 30", xlabel="x",ylabel="y")
axes[3].set_aspect('equal', adjustable='datalim')
axes[3].legend()

axes[4].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'])
axes[4].set(title="e) SG, window 4, polyorder 3", xlabel="x",ylabel="y")
axes[4].set_aspect('equal', adjustable='datalim')
axes[4].legend()

axes[5].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'])
axes[5].set(title="f) SG, window 10, polyorder 4", xlabel="x",ylabel="y")
axes[5].set_aspect('equal', adjustable='datalim')
axes[5].legend()


# Create a Rectangle patch
x_lower_lim = 260
y_lower_lim = 300
height = 100
width = 80
rect1 = patches.Rectangle((x_lower_lim, y_lower_lim), width, height, linewidth=1, edgecolor='k', facecolor='none')
rect2 = patches.Rectangle((x_lower_lim, y_lower_lim), width, height, linewidth=1, edgecolor='k', facecolor='none')


fig2, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
#fig2.suptitle("Comparison of different filters on left wrist semi-circle movement", fontsize=20)
axes = np.reshape(axes, -1)

axes[0].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw data")
axes[0].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA, window of 3")
axes[0].plot(df['left wrist x (10)'], df['left wrist y (10)'], label="MA, window of 10")
axes[0].plot(df['left wrist x (30)'], df['left wrist y (30)'], label="MA, window of 30")
axes[0].set(title="a) Moving average filters", xlabel="x",ylabel="y")
axes[0].set_aspect('equal', adjustable='datalim')
axes[0].add_patch(rect1)
axes[0].legend()

axes[1].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw")
axes[1].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'], label="SG (window 4, polyorder 3)")
axes[1].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'], label="SG (window 10, polyorder 4)")
axes[1].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA (window 3)")
axes[1].set(title="b) Savintzky-Golay filters and moving avg with window 3", xlabel="x",ylabel="y")
axes[1].set_aspect('equal', adjustable='datalim')
axes[1].add_patch(rect2)
axes[1].legend()

axes[2].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw data")
axes[2].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA, window 3")
axes[2].plot(df['left wrist x (10)'], df['left wrist y (10)'], label="MA, window 10")
axes[2].plot(df['left wrist x (30)'], df['left wrist y (30)'], label="MA, window 30")
axes[2].set(title="c) Zoomed in moving average filters", xlabel="x",ylabel="y")
axes[2].set_aspect('equal', adjustable='datalim')
axes[2].set_ylim(y_lower_lim, y_lower_lim + height)
axes[2].set_xlim(x_lower_lim, x_lower_lim + width)
axes[2].legend()

axes[3].plot(df['left wrist x raw'], df['left wrist y raw'], label="raw")
axes[3].plot(df['left wrist x (SG w4 p3)'], df['left wrist y (SG w4 p3)'], label="SG (window 4, polyorder 3)")
axes[3].plot(df['left wrist x (SG w10 p4)'], df['left wrist y (SG w10 p4)'], label="SG (window 10, polyorder 4)")
axes[3].plot(df['left wrist x (3)'], df['left wrist y (3)'], label="MA (window 3)")
axes[3].set(title="d) Zoomed in Savintzky-Golay filters and moving avg with window 3", xlabel="x",ylabel="y")
axes[3].set_aspect('equal', adjustable='datalim')
axes[3].set_ylim(y_lower_lim, y_lower_lim + height)
axes[3].set_xlim(x_lower_lim, x_lower_lim + width)
axes[3].legend()

axes[4].plot(df['time'], df['left wrist x raw'], label="raw data")
axes[4].plot(df['time'], df['left wrist x (3)'], label="MA, w 3")
axes[4].plot(df['time'], df['left wrist x (10)'], label="MA, w 10")
axes[4].plot(df['time'], df['left wrist x (30)'], label="MA, w 30")
axes[4].set(title="e) Moving average filter over time", xlabel="time [s]",ylabel="x coordinate")
axes[4].legend(loc="center left")

axes[5].plot(df['time'], df['left wrist x raw'], label="raw data")
axes[5].plot(df['time'], df['left wrist x (3)'], label="MA, window of 3")
axes[5].plot(df['time'], df['left wrist x (10)'], label="MA, window of 10")
axes[5].plot(df['time'], df['left wrist x (SG w4 p3)'], label="SG, w 4, p 3")
axes[5].plot(df['time'], df['left wrist x (SG w10 p4)'], label="SG, w 10, p 4")
axes[5].set(title="f) Savintzky-Golay over time", xlabel="time [s]",ylabel="x coordinate")
axes[5].legend(loc="center left")

plt.show()

print(df)