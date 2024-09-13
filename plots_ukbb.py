import datetime

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

size_dot = 5
n_samples = None
#
dir_ = "/projects/prime/ukbb/preprocessed_data_2024/imputations/ukbb_imputed_food_detailed_pm_sg_02_imputers2_pmm/"
full_source_name = dir_ + "raw_no_na.csv"

df = pd.read_table(full_source_name, nrows=n_samples, sep=',')

df["gender_colors"] = np.where(df["gender.0"] == 1, "red", "blue")
df["diabetes_colors"] = np.where(df["Diabetes_2.0"] == 1, "red", "green")

df["visit.0"] = pd.to_datetime(df['date_visit.0'])
df["visit.2"] = pd.to_datetime(df['date_visit.2'])
df["time_difference_days.0.2"] = (df["visit.2"] - df["visit.0"]).dt.days
df["time_difference_years.0.2"] = df["time_difference_days.0.2"]/365.25

df["age_years.2"] = df["age_years.0"] + df["time_difference_years.0.2"]

mean_value = df['time_difference_days.0.2'].mean()
std_dev = df['time_difference_days.0.2'].std()
df["time_difference_days.0.2"] = (df["time_difference_days.0.2"] - mean_value) / std_dev
''' 
plt.scatter(df["num_mem_maximum_digits_remembered_correctly.0"], df["num_mem_maximum_digits_remembered_correctly.2"], s=size_dot, c=df["diabetes_colors"])
plt.xlabel("num_mem_maximum_digits_remembered_correctly.0")
plt.ylabel("num_mem_maximum_digits_remembered_correctly.2")
plt.title('Numeric memory visit 2 vs baseline')

plt.scatter(df["snap_game_true_pos_rt_avrg.0"], df["snap_game_true_pos_rt_avrg.2"], s=size_dot, c=df["HbA1c.0"], cmap='viridis')
plt.xlabel("snap_game_true_pos_rt_avrg.0")
plt.ylabel("snap_game_true_pos_rt_avrg.2")
plt.title('Snap game visit 2 vs baseline')
#plt.satter(df["age_years.0"], df["snap_game_true_pos_rt_avrg.progression.median.0.2"], s=size_dot, c="blue")
#plt.scatter(df["HbA1c.0"], df["pairs_matching_sum_incorrect.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
plt.show()

exit(0)
'''
df["snap_game_true_pos_rt_avrg.progression.0.2"] = \
    (df["snap_game_true_pos_rt_avrg.2"] - df["snap_game_true_pos_rt_avrg.0"]) * (-1)

df["pairs_matching_sum_incorrect.progression.0.2"] = \
    (df["pairs_matching_sum_incorrect.2"] - df["pairs_matching_sum_incorrect.0"]) * (-1)

for col in df.columns:
    value = df[col][0]
    if isinstance(value, datetime.datetime) or isinstance(value, str):
        continue
    print(col)
    print(value)
    mean_value = df[col].mean()
    std_dev = df[col].std()
    df[col] = (df[col] - mean_value) / std_dev





fig, axs = plt.subplots(4, 2, figsize=(20, 15))

axs[0,0].scatter(df["age_years.0"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[0,0].set_xlabel("age_years.0")

axs[1,0].scatter(df["time_difference_days.0.2"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[1,0].set_xlabel("time_difference_days.0.2")

axs[2,0].scatter(df["HbA1c.0"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[2,0].set_xlabel("HbA1c.0")

axs[3,0].scatter(df["years_education.0"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[3,0].set_xlabel("years_education.0")

axs[0,1].scatter(df["age_years.0"], df["pairs_matching_sum_incorrect.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[0,1].set_xlabel("age_years.0")

axs[1,1].scatter(df["time_difference_days.0.2"], df["pairs_matching_sum_incorrect.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[1,1].set_xlabel("time_difference_days.0.2")

axs[2,1].scatter(df["HbA1c.0"], df["pairs_matching_sum_incorrect.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[2,1].set_xlabel("HbA1c.0")

axs[3,1].scatter(df["years_education.0"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["diabetes_colors"])
axs[3,1].set_xlabel("years_education.0")

'''fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["HbA1c.0"], df["hdl.0"], df["snap_game_true_pos_rt_avrg.progression.0.2"], s=size_dot, c=df["snap_game_true_pos_rt_avrg.progression.0.2"], cmap='viridis')
ax.set_xlabel('HbA1c.0')
ax.set_ylabel('hdl.0')
ax.set_zlabel('snap_game_true_pos_rt_avrg.progression.0.2')
ax.view_init(elev=0, azim=45)
'''
plt.show()

# Add a title for the whole figure
# fig.suptitle(data_filename)


