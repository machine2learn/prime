import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')
matplotlib.use('module://ipykernel.pylab.backend_inline')
import matplotlib.pyplot as plt
import math

fld_age_0 = "f.21022.0.0"
field_id = "f.eid"

codes = {
    "snap_game": { # no abandon button
        "ukbb_ms_to_first_press": 404,
        "ukbb_index_card_a": 401,
        "ukbb_index_card_b": 402,
        "ukbb_n_button_presses": 403,
        "ukbb_mean_rt": 20023,
        "instances": [0, 1, 2],
        "training_rounds": [0, 1, 2, 3, 4],
        "n_rounds": 12,
        "out_names": ["n_true_positives", "n_true_negatives", "n_false_positives", "n_false_negatives",
                      "snap_game_true_pos_rt_avrg", "snap_game_false_pos_rt_avrg", "snap_game_true_pos_rt_std",
                      "snap_game_false_pos_rt_std"]
    },
    "numeric_memory": { # -1 abandoned; the test available for a subset of approx 52 00 participants, so nan is nan
        "ukbb_target_numer_to_enter": 4252,
        "ukbb_entered_number": 4258,
        "ukbb_ms_to_first_key": 4254,
        "ukbb_ms_to_last_key": 4255,
        "ukbb_round_nr": 4260,
        "ukbb_number_of_rounds": 4283,
        "instances": [0, 2],
        "ukbb_maximum_digits_remembered_correctly": 4282,
        "ukbb_completion_time": 4285,
        "training_rounds": [],
        "out_names": ["maximum_digits_remembered_correctly", "numeric_memory_test_abandoned"]
    },

    #  Multiple rounds were conducted. The first round used 3 pairs of cards and the second 6 pairs of cards.
    #  # In the pilot phase an additional (i.e. third) round was conducted using 6 pairs of cards. however this was
    #  dropped from the main study as the extra set of results were very similar to the second and
    #  not felt to add significant new information.
    "pairs_matching": { #abandon indicated when completion time is zero
        "ukbb_number_correct_in_round": 398,
        "ukbb_number_incorrect_in_round": 399,
        "ukbb_completion_time_of_round": 400,
        "training_rounds": [],
        "instances": [0, 1, 2],
         "rounds" : [1, 2],
        "out_names": ["pairs_matching_sum_correct", "pairs_matching_sum_incorrect", "pairs_matching_sum_time.",
                      "pairs_matching_test_abandoned"]
    }
}

map_ukbb_freindly_name = {
    20023: "snap_game_true_pos_rt_avrg",
    401: "snap_game_index_card_a",
    402: "snap_game_index_card_b",
    403: "snap_game_number_button_presses",
    404: "snap_game_ms_first_press",
    399: "pairs_matching_incorrect_in_round",
    400: "pairs_matching_completion_time_round",
    4282: "num_memory_max_digits_remembered_correctly",
    4285: "num_memory_completion_time",
    53: "date_visit",
    21022: "age_visit"
}

if __name__ == '__main__':


    n_samples = 10000# limit #samples for debug
    data_filename = "/projects/prime/ukbb/17_01_2024_selected.tab"
    output_dir = "/projects/prime/ukbb/preprocessed_data_2024/"
    out_file = output_dir + "cognitive_2024_test.csv"

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw6244_207_20231205.txt'
    # data_filename = '~/PRIME/python_raw_cognitive_tmp_distorted.csv'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    ### find out the list of all columns
    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    # make a list of columns needed
    needed_fields = ["f.eid"]
    for ukbb_field in map_ukbb_freindly_name:
        print(ukbb_field)
        current_list = [col for col in data_columns if col.startswith("f." +str(ukbb_field))]
        needed_fields = needed_fields + current_list


    print("start reading the table .... ")

    df = pd.read_table(data_filename, sep='\t', usecols=needed_fields, dtype=str, nrows=n_samples)
    # print(df['f.41281.0.46'])
    # exit()

    # df.to_csv('/projects/prime/ukbb/python_raw_cognitive_tmp.csv', sep=';')

    print("finished reading the table .... ")
    print("# participants before checking entrance age")
    print(len(df))
    df = df[~df[fld_age_0].isnull()]
    print("# participants after checking entrance age but yet with withdrawn ")
    print(len(df))

    df = df[~df[field_id].isin(withdrawn_col)]

    print("# participants without withdrawn ")
    print(len(df))

    # loop over all the columns to give them freindly names
    for i in range(4):

        for ukbb_field in map_ukbb_freindly_name:
            prefix = f"f.{str(ukbb_field)}.{i}"
            current_array = [col for col in needed_fields if col.startswith(prefix)] # should be all 1 except pm
            print(map_ukbb_freindly_name[ukbb_field])
            print(current_array)
            print("***")
            if map_ukbb_freindly_name[ukbb_field].startswith("pairs_matching_incorrect_in_round"):
                df["pairs_matching_sum_incorrect."+str(i)] = 0
            for j in range(len(current_array)+1):
                clmn = prefix + "." + str(j)
                if clmn in current_array:
                    if not map_ukbb_freindly_name[ukbb_field].startswith("date_visit"):
                        df[clmn] = df[clmn].astype(float)
                    if map_ukbb_freindly_name[ukbb_field].startswith("pairs_matching"):
                        friendly_name = map_ukbb_freindly_name[ukbb_field] + "_round_" + str(j) + "." +str(i)
                        friendly_name = f"{map_ukbb_freindly_name[ukbb_field]}_round_{j}.{i}"
                        df["pairs_matching_sum_incorrect." + str(i)] = \
                            df["pairs_matching_sum_incorrect."+str(i)] + df[clmn]
                    else:
                        friendly_name = f"{map_ukbb_freindly_name[ukbb_field]}.{i}"
                    df.rename(columns={clmn: friendly_name}, inplace=True)

    print(df.columns)

    df["time_difference.0.2"] = (pd.to_datetime(df["date_visit.2"]) - pd.to_datetime(df["date_visit.0"])).dt.total_seconds() / (24 * 3600)
    df["time_difference.0.1"] = (pd.to_datetime(df["date_visit.1"]) - pd.to_datetime(df["date_visit.0"])).dt.total_seconds() / (24 * 3600)
    df["num_memory_max_digits_remembered_correctly.0"]

    df["num_memory_max_digits_remembered_correctly.0"].replace(-1,0, inplace = True)
    df["num_memory_max_digits_remembered_correctly.2"].replace(-1, 0, inplace=True)
    df["num_memory_max_digits_remembered_correctly.3"].replace(-1, 0, inplace=True)

    df_checkup_nm = df[~df["num_memory_max_digits_remembered_correctly.0"].isnull() &
                       ~df["num_memory_max_digits_remembered_correctly.2"].isnull()]
    df_checkup_nm_sg = df_checkup_nm[~df_checkup_nm["snap_game_true_pos_rt_avrg.0"].isnull() &
                                     ~df_checkup_nm["snap_game_true_pos_rt_avrg.2"].isnull()]

    # plt.figure(figsize=(12, 5))

    # Boxplot
    # plt.subplot(1, 2, 1)
    # plt.boxplot(df_checkup_nm_sg["time_difference.0.2"])
    # plt.title('Boxplot of time difference visit 0 and 2')

    # Histogram
    # plt.subplot(1, 2, 2)
    df_checkup_nm_sg["time_difference.0.2"].plot(kind='hist', title='Density time_difference.0.2 for all nm, sg')
    plt.show()

    df_checkup_nm_sg_3500_3750 = df_checkup_nm_sg[df_checkup_nm_sg["time_difference.0.2"].ge(3500) &
                                                  df_checkup_nm_sg["time_difference.0.2"].le(3750)]
    df_checkup_nm_sg_3500_3750["time_difference.0.2"].plot(kind='hist', title='Density time_difference.0.2')
    plt.show()

    df_checkup_nm_sg_3500_3750.loc[:, "numeric_memory_change.0.2"] = \
        df_checkup_nm_sg_3500_3750["num_memory_max_digits_remembered_correctly.2"] - df_checkup_nm_sg_3500_3750["num_memory_max_digits_remembered_correctly.0"]

    print(f"sample size {len(df_checkup_nm_sg_3500_3750)}")
    min_age = math.floor(np.min(df_checkup_nm_sg_3500_3750["age_visit.0"]))
    max_age = math.ceil(np.max(df_checkup_nm_sg_3500_3750["age_visit.0"]))
    n_rows = max_age - min_age + 1
    row_list = []

    for age_years in range(min_age, max_age+1):
        print(age_years)
        df_current = df_checkup_nm_sg_3500_3750[df_checkup_nm_sg_3500_3750["age_visit.0"].ge(min_age) &
                                                 df_checkup_nm_sg_3500_3750["age_visit.0"].le(age_years)]
        if len(df_current) == 0:
            continue
        row_list.append([age_years, np.median(df_current["numeric_memory_change.0.2"])])

    matrix_median = np.array(row_list)
    print(matrix_median)
    # plt.scatter(df_checkup_nm_sg_3500_3750["age_visit.0"], df_checkup_nm_sg_3500_3750["numeric_memory_change.0.2"])
    plt.plot(matrix_median[:,0], matrix_median[:,1])
    plt.show()
    exit(0)

    # difference visits fields

    # statistics

    # derived fields

    df["pairs_matching_sum_incorrect"]
    # print(df[["f.404.0.0", "f.404.0.1"]])
    df_num_mem_mon_1_checkup = df[df["f.4282.0.0"].astype(float).eq(-1)]
    print(f"# of ppl -1 of visit 0 {len(df_num_mem_mon_1_checkup)}")
    df_num_mem_completed = df[~df["f.4282.0.0"].isnull() & ~df["f.4282.2.0"].isnull()]
    print(f"# of ppl completed num memory visit 0 and visit 2{len(df_num_mem_completed)}")
    df_num_mem_completedion_time = df[~df["f.4285.0.0"].isnull() & ~df["f.4285.2.0"].isnull()]
    print(f"# of ppl completion time  visit 0 and visit 2{len(df_num_mem_completedion_time)}")
    df_num_mem_abandoned_0 = df_num_mem_completed[df_num_mem_completed["f.4282.0.0"].astype(float).eq(-1)]
    print(f"# of ppl abandoned in visit 0 from those who registered for both visits num mem  {len(df_num_mem_abandoned_0 )}")
    df_num_mem_abandoned_2 = df_num_mem_completed[df_num_mem_completed["f.4282.2.0"].astype(float).eq(-1)]
    print(f"# of ppl abandoned in visit 2 from those who registered for both visits num mem  {len(df_num_mem_abandoned_2)}")
    df_num_mem_abandoned_0_2 =df_num_mem_abandoned_0[df_num_mem_abandoned_0["f.4282.2.0"].astype(float).eq(-1)]
    print(f"# of ppl abandoned in visit 0 AND 2 from those who registered for both visits num mem  {len(df_num_mem_abandoned_0_2)}")

    df_num_mem_snap_game_completed = df_num_mem_completed[~df_num_mem_completed["f.20023.0.0"].isnull() &
                                                          ~df_num_mem_completed["f.20023.2.0"].isnull()]

    print(
        f"# of ppl completed snap game and mum memory {len(df_num_mem_snap_game_completed)}")

    num_mem_0 = df_num_mem_snap_game_completed["f.4282.2.0"].astype(float).replace(-1,0, inplace = False)
    num_mem_2 = df_num_mem_snap_game_completed["f.4282.0.0"].astype(float).replace(-1,0, inplace = False)
    numeric_memory_change = num_mem_2 - num_mem_0
    df_num_mem_snap_game_completed["change"] = numeric_memory_change

    plt.scatter(df_num_mem_snap_game_completed["f.21022.0.0"], df_num_mem_snap_game_completed["change"])
    plt.show()
    exit(0)



    std_dev = np.std(numeric_memory_change)
    variance = np.var(numeric_memory_change)
    data_range = np.ptp(numeric_memory_change)  # Peak to peak (max-min)
    iqr = np.subtract(*np.percentile(numeric_memory_change, [75, 25]))  # 75th percentile - 25th percentile
    cv = std_dev / np.mean(numeric_memory_change)


    # Displaying the measures
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Range: {data_range}")
    print(f"IQR: {iqr}")
    print(f"Coefficient of Variation: {cv}")

    # Visualizing data variability
    plt.figure(figsize=(12, 5))

    # Boxplot
    plt.subplot(1, 2, 1)
    plt.boxplot(numeric_memory_change)
    plt.title('Boxplot of numeric_memory_change')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(numeric_memory_change, bins=20, edgecolor='k')
    plt.title('Histogram of numeric_memory_change')

    plt.tight_layout()
    plt.show()

    exit(0)
    std_dev = np.std(numeric_memory_change)
    variance = np.var(numeric_memory_change)
    data_range = np.ptp(numeric_memory_change)  # Peak to peak (max-min)
    iqr = np.subtract(*np.percentile(numeric_memory_change, [75, 25]))  # 75th percentile - 25th percentile
    cv = std_dev / np.mean(numeric_memory_change)

    # Displaying the measures
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")
    print(f"Range: {data_range}")
    print(f"IQR: {iqr}")
    print(f"Coefficient of Variation: {cv}")

    # Visualizing data variability
    plt.figure(figsize=(12, 5))

    # Boxplot
    plt.subplot(1, 2, 1)
    plt.boxplot(numeric_memory_change)
    plt.title('Boxplot of numeric_memory_change')

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(numeric_memory_change, bins=10, edgecolor='k')
    plt.title('Histogram of numeric_memory_change')

    plt.tight_layout()
    plt.show()
    df_pairs_matching_snap_game_took_part = df[~df["f.20023.0.0"].isnull() & ~df["f.20023.2.0"].isnull() &
                                               ~df["f.400.0.2"].isnull() & ~df["f.400.2.2"].isnull()]
    print(
        f"# of ppl took part snap game and pm {len(df_pairs_matching_snap_game_took_part)}")

    df_pairs_matching_snap_game_completed = \
        df_pairs_matching_snap_game_took_part[df_pairs_matching_snap_game_took_part["f.400.0.2"].astype(float).gt(0) &
                                              df_pairs_matching_snap_game_took_part["f.400.2.2"].astype(float).gt(0)]

    print(
        f"# of ppl completed pn with snap game  {len(df_pairs_matching_snap_game_completed)}")
    exit(0)



    df.to_csv(out_file, sep=',')






