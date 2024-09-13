import pandas as pd
import numpy as np

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

if __name__ == '__main__':

    n_samples = 1000 # limit #samples for debug
    data_filename = "/projects/prime/ukbb/17_01_2024_selected.tab"
    output_dir = "/projects/prime/ukbb/preprocessed_data_2024/"
    out_file = output_dir + "cognitive_2024_test.csv"

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw6244_207_20231205.txt'
    # data_filename = '~/PRIME/python_raw_cognitive_tmp_distorted.csv'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    ### find out the list of all columns
    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    fields_visit_dates = [col for col in data_columns if col.startswith("f.53.")]
    # fields_visit_ages = [col for col in data_columns if col.startswith("f.21003.")]

    all_fields_ = [field_id] + [fld_age_0] + fields_visit_dates
    for test_name in codes:
        print(test_name)
        for key in codes[test_name]:
            if key.startswith("ukbb_"):
                print(key + "_" + str(codes[test_name][key]))
                current_list = [col for col in data_columns if col.startswith("f." + str(codes[test_name][key]))]
                print(current_list)
                all_fields_ = all_fields_ + current_list


    print("start reading the table .... ")

    df = pd.read_table(data_filename, sep='\t', usecols=all_fields_, dtype=str, nrows=n_samples)
    # print(df['f.41281.0.46'])
    # exit()

    # df.to_csv('/projects/prime/ukbb/python_raw_cognitive_tmp.csv', sep=';')

    print("finished reading the table .... ")
    print("# participants before checking entrance age")
    print(len(df))
    df_ = df[~df[fld_age_0].isnull()]
    df = df_
    print("# participants after checking entrance age but yet with withdrawn ")
    print(len(df))

    ones = pd.Series(int(1), index=df.index, dtype=int)

    df = df[~df[field_id].isin(withdrawn_col)]

    print("# participants without withdrawn ")
    print(len(df))
    # print(df[["f.404.0.0", "f.404.0.1"]])

    ## check on numeric memory


    for test_name in codes:
        for fld_name in codes[test_name]:
            print(f"replacing {codes[test_name][fld_name]} with {fld_name}")
            if fld_name.startswith("ukbb_"):
                current_list = [col for col in data_columns if col.startswith("f." + str(codes[test_name][fld_name]))]
                for clmn in current_list:
                    print(clmn)
                    convenient_name = test_name + "_" + clmn.replace(str(codes[test_name][fld_name]), fld_name)
                    df.rename(columns={clmn: convenient_name}, inplace=True)

    convenient_fields_visit_dates = []
    for dt_clmn in fields_visit_dates:
        convenient_name = dt_clmn.replace("f.53", "ukbb_date_visit")
        convenient_fields_visit_dates = convenient_fields_visit_dates + [convenient_name]
        df.rename(columns={dt_clmn: convenient_name}, inplace=True)

    # snap game

    count_sg_0 = df['snap_game_f.ukbb_mean_rt.0.0'].astype("float").ge(0).sum()
    print(f"snap game: # who did baseline test {count_sg_0}")
    count_sg_2 = df['snap_game_f.ukbb_mean_rt.2.0'].astype("float").ge(0).sum()
    print(f"snap game: # who did the second visit {count_sg_2}")

    print("snap game")
    prefix_a = "snap_game_f.ukbb_index_card_a."
    prefix_b = "snap_game_f.ukbb_index_card_b."
    prefix_rt = "snap_game_f.ukbb_ms_to_first_press."
    prefix_bp = "snap_game_f.ukbb_n_button_presses."

    problematic_participants = []

    for instance in codes["snap_game"]["instances"]:

        print(f"instance:{instance}")

        fld_time = "ukbb_date_visit." + str(instance) + ".0"
        present_at_visit = ~df[fld_time].isnull()


        new_fld_tp = "snap_game_n_true_positives."+str(instance)
        new_fld_tn = "snap_game_n_true_negatives." + str(instance)
        new_fld_fp = "snap_game_n_false_positives." + str(instance)
        new_fld_fn = "snap_game_n_false_negatives." + str(instance)

        new_fld_avrg_rt_tp = "snap_game_true_pos_rt_avrg."+str(instance)
        new_fld_avrg_rt_fp = "snap_game_false_pos_rt_avrg." + str(instance)

        new_fld_std_rt_tp = "snap_game_true_pos_rt_std." + str(instance)
        new_fld_std_rt_fp = "snap_game_false_pos_rt_std." + str(instance)

        df.insert(1, new_fld_std_rt_fp, pd.Series(None, index=df.index, dtype=float))
        df.insert(1, new_fld_std_rt_tp, pd.Series(None, index=df.index, dtype=float))

        df.insert(1, new_fld_avrg_rt_fp, pd.Series(None, index=df.index, dtype=float))
        df.insert(1, new_fld_avrg_rt_tp, pd.Series(None, index=df.index, dtype=float))

        df.insert(1, new_fld_fn, pd.Series(int(0), index=df.index, dtype=int))
        df.insert(1, new_fld_fp, pd.Series(int(0), index=df.index, dtype=int))
        df.insert(1, new_fld_tn, pd.Series(int(0), index=df.index, dtype=int))
        df.insert(1, new_fld_tp, pd.Series(int(0), index=df.index, dtype=int))

        list_rt_columns_tp = []
        list_rt_columns_fp = []


        for round_ in range(codes["snap_game"]["n_rounds"]):
            if round_ in codes["snap_game"]["training_rounds"]:
                continue

            print(f"round:{round_}")
            fld_a = prefix_a + str(instance) + "." + str(round_)
            fld_b = prefix_b + str(instance) + "." + str(round_)
            fld_rt = prefix_rt + str(instance) + "." + str(round_)
            fld_bp = prefix_bp + str(instance) + "." + str(round_)

            fld_tp_rt = "tp_" + prefix_rt + str(instance) + "." + str(round_)
            fld_fp_rt = "fp_" + prefix_rt + str(instance) + "." + str(round_)

            df.insert(1, fld_tp_rt, pd.Series(None, index=df.index, dtype=float))
            df.insert(1, fld_fp_rt, pd.Series(None, index=df.index, dtype=float))

            bp = df[fld_bp].astype(float).ge(1)

            true_positives_in_round = present_at_visit & df[fld_a].eq(df[fld_b]) & bp
            true_negatives_in_round = present_at_visit & df[fld_a].ne(df[fld_b]) & ~bp
            false_positives_in_round = present_at_visit & df[fld_a].ne(df[fld_b]) & bp
            false_negatives_in_round = present_at_visit & df[fld_a].eq(df[fld_b]) & ~bp

            df.loc[true_positives_in_round, fld_tp_rt] = df[fld_rt]
            df.loc[false_positives_in_round, fld_fp_rt] = df[fld_rt]

            df.loc[true_positives_in_round, new_fld_tp] = df[new_fld_tp] + true_positives_in_round.astype(int)
            df.loc[true_negatives_in_round, new_fld_tn] = df[new_fld_tn] + true_negatives_in_round.astype(int)
            df.loc[false_positives_in_round, new_fld_fp] = df[new_fld_fp] + false_positives_in_round.astype(int)
            df.loc[false_negatives_in_round, new_fld_fn] = df[new_fld_fn] + false_negatives_in_round.astype(int)

            list_rt_columns_tp = list_rt_columns_tp + [fld_tp_rt]
            list_rt_columns_fp = list_rt_columns_fp + [fld_fp_rt]

            # round is finished
        df.loc[~present_at_visit, new_fld_tp] = None
        df.loc[~present_at_visit, new_fld_tn] = None
        df.loc[~present_at_visit, new_fld_fp] = None
        df.loc[~present_at_visit, new_fld_fn] = None

        # taking averages of rt
        df.loc[present_at_visit, new_fld_avrg_rt_tp] = df[list_rt_columns_tp].astype("float").mean(axis=1, skipna=True)
        df.loc[present_at_visit, new_fld_avrg_rt_fp] = df[list_rt_columns_fp].astype("float").mean(axis=1, skipna=True)

        # taking std of rt
        df.loc[present_at_visit, new_fld_std_rt_tp] = df[list_rt_columns_tp].astype("float").std(axis=1, skipna=True)
        df.loc[present_at_visit,  new_fld_std_rt_fp] = df[list_rt_columns_fp].astype("float").std(axis=1, skipna=True)

    print("sanity check")
    # sanity check
    for intsance in codes["snap_game"]["instances"]:
        df['check_sum'] = df["snap_game_n_true_positives." + str(instance)] + \
                  df["snap_game_n_true_negatives." + str(instance)] + \
                    df["snap_game_n_false_positives." + str(instance)] + \
                    df["snap_game_n_false_negatives." + str(instance)]
        test_n = 12-5

        critera_column = (df['check_sum'].eq(test_n) | df['check_sum'].isnull())
        if not critera_column.all():
            print(str(instance))
            problematic_df = df[critera_column]
            problematic_df.to_csv('/projects/prime/ukbb/python_preprocessed_cognitive_failed_checksum_sg.csv', sep=';')
            exit("sanity check snap game fails, in some runs the sum of responses is not equal "
                 "to the amount of stimuli, check the error file")

        mean_fld_computed = "snap_game_true_pos_rt_avrg."+str(instance)
        mean_fld_ukbb = "snap_game_f.ukbb_mean_rt."+str(instance)+".0"
        tolerance = 0.5
        df['check_mean'] = ((df[mean_fld_ukbb].astype(float).notna() | df[mean_fld_computed].notna()) &
             (df[mean_fld_ukbb].astype(float) - df[mean_fld_computed]).abs().le(tolerance)) | \
                           (df[mean_fld_ukbb].isnull() & df[mean_fld_computed].isnull())
        if not df['check_mean'].all():
            print(str(instance))
            print(df[[mean_fld_ukbb, mean_fld_computed]])
            problematic_df = df[~df['check_mean']]
            problematic_df.to_csv('/projects/prime/ukbb/python_preprocessed_cognitive_failed_checksum_sg.csv', sep=';')
            print("WARNING!:Check the error files when working with the snap game test! "
                  "For some participants the discrepance "
                  "between computed and ukbb-supplied mean for true positive RT is too big. "
                  "I exclude these participants from the output file, if the discrepance is in instances 0-2.")
            if instance < 3:
                current_problematic_participants = df.loc[~df['check_mean'], "f.eid"].tolist()
                problematic_participants = problematic_participants + current_problematic_participants

    df['snap_game_instance_comparison'] = df["snap_game_f.ukbb_mean_rt.0.0"].isnull() & \
                                          df["snap_game_f.ukbb_mean_rt.2.0"].notna()
    count_no_baseline_sg = df['snap_game_instance_comparison'].sum()
    print(f"snap game: # who did the second withit without baseline test {count_no_baseline_sg}")

    # num memory, it looks like no processing is needed

    prefix_max_digits = "numeric_memory_f.ukbb_maximum_digits_remembered_correctly."
    print("numeric memory")
    for instance in codes["numeric_memory"]["instances"]:
        print("Visit "+str(instance))
        fld_time = "ukbb_date_visit." + str(instance) + ".0"
        present_at_visit = ~df[fld_time].isnull()
        new_fld_test_abandoned = "numeric_memory_test_abandoned." + str(instance)
        df.insert(1, new_fld_test_abandoned, pd.Series(None, index=df.index, dtype=float))
        fld_max_digits = prefix_max_digits + str(instance) + ".0"
        df.loc[present_at_visit & df[fld_max_digits].astype(float).eq(-1), new_fld_test_abandoned] = 1
        df.loc[present_at_visit & df[fld_max_digits].astype(float).ge(0), new_fld_test_abandoned] = 0
        cca_friendly_fld_name = "num_mem_maximum_digits_remembered_correctly." + str(instance)
        df.insert(1, cca_friendly_fld_name, df[fld_max_digits])
        df.loc[present_at_visit & df[cca_friendly_fld_name].astype(float).eq(-1), cca_friendly_fld_name] = 0


    # "ukbb_number_correct_in_round": 398,
    # "ukbb_number_incorrect_in_round": 399,
    # "ukbb_completion_time_of_round": 400,

    print("pairs matching")

    count_pm_0 = df['pairs_matching_f.ukbb_completion_time_of_round.0.1'].astype("float").ge(0).sum()
    print(f"pm: # who did baseline test round 0  {count_pm_0}")
    count_pm_02 = df['pairs_matching_f.ukbb_completion_time_of_round.0.2'].astype("float").ge(0).sum()
    print(f"pm: # who did baseline test round 2  {count_pm_02}")
    count_pm_2 = df['pairs_matching_f.ukbb_completion_time_of_round.2.1'].astype("float").ge(0).sum()
    print(f"pm: # who did the second visit round 1 {count_pm_2}")
    count_pm_22 = df['pairs_matching_f.ukbb_completion_time_of_round.2.2'].astype("float").ge(0).sum()
    print(f"pm: # who did the second visit round 2 {count_pm_22}")

    prefix_correct = "pairs_matching_f.ukbb_number_correct_in_round."
    prefix_incorrect = "pairs_matching_f.ukbb_number_incorrect_in_round."
    prefix_completion_time = "pairs_matching_f.ukbb_completion_time_of_round."

    for instance in codes["pairs_matching"]["instances"]:

        print(f"instance:{instance}")

        fld_time = "ukbb_date_visit." + str(instance) + ".0"
        df['present_at_visit'] = ~df[fld_time].isnull()

        new_fld_sum_correct = "pairs_matching_sum_correct." + str(instance)
        new_fld_sum_incorrect = "pairs_matching_sum_incorrect." + str(instance)
        new_fld_sum_time = "pairs_matching_sum_time." + str(instance)
        new_test_abandoned = "pairs_matching_test_abandoned." + str(instance)

        df.insert(1, new_test_abandoned, pd.Series(False, index=df.index, dtype=bool))
        df.insert(1, new_fld_sum_correct, pd.Series(int(0), index=df.index))
        df.insert(1, new_fld_sum_incorrect, pd.Series(int(0), index=df.index))
        df.insert(1, new_fld_sum_time, pd.Series(float(0), index=df.index))

        for round_ in codes["pairs_matching"]["rounds"]:

            print(f"round:{round_}")

            fld_completion_time = prefix_completion_time + str(instance) + "." + str(round_)
            df.loc[df[fld_completion_time].astype(float).eq(0.0), new_test_abandoned] = True
            df.loc[df[fld_completion_time].isnull(), new_test_abandoned] = np.nan

            df.loc[df[new_test_abandoned].eq(False), new_fld_sum_time] = \
                df[new_fld_sum_time] + df[fld_completion_time].astype(float)
            df.loc[df[new_test_abandoned].ne(False), new_fld_sum_time] = np.nan

            fld_correct = prefix_correct + str(instance) + "." + str(round_)
            df.loc[df[new_test_abandoned].eq(False), new_fld_sum_correct] = \
                df[new_fld_sum_correct] + df[fld_correct].astype(float)
            df.loc[df[new_test_abandoned].ne(False), new_fld_sum_correct] = np.nan

            fld_incorrect = prefix_incorrect + str(instance) + "." + str(round_)
            df.loc[df[new_test_abandoned].eq(False), new_fld_sum_incorrect] = \
                df[new_fld_sum_incorrect] + df[fld_incorrect].astype(float)
            df.loc[df[new_test_abandoned].ne(False), new_fld_sum_incorrect] = np.nan

            # sanity check: in round 1 there must be 3 pairs, adn and round2 must be 6 pairs (if not abandoned

            if round_ == 1:
                check = (df[new_fld_sum_correct].isnull() | df[new_fld_sum_correct].eq(3)).all()
                if not check:
                    print("Pairs matching, round 1, visit:")
                    print(str(instance))
                    problematic_df = df[df[new_fld_sum_correct].isnull() & df[new_fld_sum_correct].ne(3)]
                    problematic_df.to_csv('/projects/prime/ukbb/python_preprocessed_cognitive_failed_checksum_1_pm.csv',
                                          sep=';')
                    exit("sanity check pairs matching fails, in the first round there are participants "
                         "with != 3 correct answers")

            if round_ == 2:
                check = (df[new_fld_sum_correct].isnull() | df[new_fld_sum_correct].eq(9)).all()
                if not check:
                    print("Pairs matching, round 2, visit:")
                    print(str(instance))
                    problematic_df =df[df[new_fld_sum_correct].isnull() & df[new_fld_sum_correct].ne(9)]
                    problematic_df.to_csv('/projects/prime/ukbb/python_preprocessed_cognitive_failed_checksum_2_pm.csv',
                                          sep=';')
                    exit("sanity check pairs matching fails, in the whole test there are participants "
                         "with != 9 correct answers")

    df['pm_instance_comparison'] = df["pairs_matching_f.ukbb_number_incorrect_in_round.0.1"].isnull() & \
                                          df["pairs_matching_f.ukbb_number_incorrect_in_round.2.1"].notna()
    count_no_baseline_pm = df['pm_instance_comparison'].sum()
    print(f"pm: # who did the second withit without baseline test {count_no_baseline_pm}")

    new_data_columns = df.columns
    output_columns = ["f.eid"] + convenient_fields_visit_dates
    for key in codes:
        for criterion in codes[key]["out_names"]:
            output_columns = output_columns + [col for col in new_data_columns if criterion in col]

    print(output_columns)

    problematic_participants = np.unique(np.array(problematic_participants))
    df = df[~df["f.eid"].isin(problematic_participants)]
    pp = len(problematic_participants)
    if pp > 0:
        print("There were participants for which some sanity check failed, please look at the warning messages and "
              "Check the files with 'failed' in their names.")
        print(f"These participants are excluded from the final report for now. There are  {pp} such participants.")

    out_file = output_dir + "/cognitive_2024.csv"
    res = df[output_columns]

    n_num_memory = (df['num_mem_maximum_digits_remembered_correctly.0'].notnull() &
                    df['num_mem_maximum_digits_remembered_correctly.2'].notnull()).sum()
    print(f"the number of participants completed num memory 0 and 2 {n_num_memory}")

    n_pairs_matching = (
                df['pairs_matching_sum_incorrect.0'].notnull() & df['pairs_matching_sum_incorrect.2'].notnull()).sum()
    print(f"the number of participants completed pairs matching 0 and 2  {n_pairs_matching}")

    res.to_csv(out_file, sep=',')






