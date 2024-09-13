import pandas as pd
import numpy as np

fld_age_0 = "f.21022.0.0"
field_id = "f.eid"


codes = {

    "online_numeric_memory": { # -1 in 20240 abandoned; the test available for a subset of approx 52 00 participants, so nan is nan
        "ukbb_test_completed": 20138,
        "ukbb_maximum_digits_remembered_correctly": 20240
    },
    # Either two or three rounds were conducted. The first round used 3 pairs of cards, the second 6 pairs of cards
    # and the third 8 pairs of cards. Participants were only presented with the third round if they made 0 or 1
    # errors on the second round.

# TODO: question how many rounds? 1, 2 or 3d as well? then the third one will have potentially more errors
    "online_pairs_matching": { #abandon indicated when completion time is zero
        "ukbb_number_incorrect_in_round": 20132,
        "ukbb_test_completed": 20134,
         "instances": [0],
        "rounds": [0, 1],
    }
}

if __name__ == '__main__':

    n_samples = None # limit samples for debug
    data_filename = "/projects/prime/ukbb/23_01_2024_selected.tab"
    output_dir = "/projects/prime/ukbb/preprocessed_data_2024/"

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw6244_207_20231205.txt'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    ### find out the list of all columns
    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    all_fields_ = [field_id] + [fld_age_0]
    for test_name in codes:
        print(test_name)
        for key in codes[test_name]:
            if key.startswith("ukbb_"):
                print(key + ": " + str(codes[test_name][key]))
                current_list = [col for col in data_columns if col.startswith("f." + str(codes[test_name][key]))]
                print(current_list)
                all_fields_ = all_fields_ + current_list


    print("start reading the table .... ")
    df = pd.read_table(data_filename, sep='\t', usecols=all_fields_, dtype=str, nrows=n_samples)

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

    for test_name in codes:
        for fld_name in codes[test_name]:
            print(f"replacing {codes[test_name][fld_name]} with {fld_name}")
            if fld_name.startswith("ukbb_"):
                current_list = [col for col in data_columns if col.startswith("f." + str(codes[test_name][fld_name]))]
                for clmn in current_list:
                    print(clmn)
                    convenient_name = test_name + "_" + clmn.replace(str(codes[test_name][fld_name]), fld_name)
                    df.rename(columns={clmn: convenient_name}, inplace=True)


    # num memory, it looks like no processing is needed
    print("online_numeric memory")
    new_fld_test_abandoned = "online_numeric_memory_test_abandoned.2"
    df.insert(1, new_fld_test_abandoned, pd.Series(None, index=df.index, dtype=float))
    fld_max_digits = "online_numeric_memory_f.ukbb_maximum_digits_remembered_correctly.0.0"
    df.loc[df[fld_max_digits].astype(float).eq(-1), new_fld_test_abandoned] = 1
    df.loc[df[fld_max_digits].astype(float).ge(0), new_fld_test_abandoned] = 0
    friendly_fld_name = "online_num_mem_maximum_digits_remembered_correctly.2"
    df.insert(1, friendly_fld_name, df[fld_max_digits])
    df.loc[df[friendly_fld_name].astype(float).eq(-1), friendly_fld_name] = 0

    # pairs matching
    print("online_pairs matching")

    prefix_incorrect = "online_pairs_matching_f.ukbb_number_incorrect_in_round.0."
    fld_completion_time = "online_pairs_matching_f.ukbb_test_completed.0.0"

    new_fld_sum_incorrect = "online_pairs_matching_sum_incorrect.2"
    df.insert(1, new_fld_sum_incorrect, pd.Series(int(0), index=df.index))

    for round_ in codes["online_pairs_matching"]["rounds"]:

        print(f"round:{round_}")

        fld_incorrect = prefix_incorrect + str(round_)
        df.loc[~df[fld_completion_time].isnull(), new_fld_sum_incorrect] = df[new_fld_sum_incorrect] +\
                                                                           df[fld_incorrect].astype(float)

    df.rename(columns={"online_pairs_matching_f.ukbb_test_completed.0.0":
                           "online_pairs_matching_test_completed_online.2"}, inplace=True)
    df.rename(columns={"online_numeric_memory_f.ukbb_test_completed.0.0":
                           "online_numeric_memory_test_completed_online.2"},
              inplace=True)
    df.rename(columns={"online_numeric_memory_f.ukbb_maximum_digits_remembered_correctly.0.0":
                           "online_numeric_memory_f.ukbb_maximum_digits_remembered_correctly_online.2"},
              inplace=True)

    df.loc[df['online_pairs_matching_test_completed_online.2'].isnull(), 'online_pairs_matching_sum_incorrect.2'] = None

    n_num_memory = df['online_numeric_memory_test_abandoned.2'].eq(0).sum()
    print(f"the number of participants completed (not abandoned) online follow up num memory {n_num_memory}")

    n_pairs_matching = df['online_pairs_matching_test_completed_online.2'].notnull().sum()
    print(f"the number of participants completed online follow up pairs matching {n_pairs_matching }")

    out_file = output_dir + "/online_cognitive_2024.csv"
    df.to_csv(out_file, sep=',')






