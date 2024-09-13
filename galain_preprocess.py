import pandas as pd
import json

n_samples = 1000

# remark: age and date of visits we get from python preprocess diagnoses
configuration_path = "../configurations/baseline_one_to_one_latency.json"
config = None
with open(configuration_path, 'r') as config_str:
  config = json.load(config_str)

input_files = config['input_files']
seps_input_files = config['seps_input_files']
factors_variables_map = config['factors_variables_map']
output_data_file_name = config['output_data_file_name']
output_factor_file_name = config['output_factor_file_name']
one_to_one = config['one_to_one']
all_defined = config['all_defined']

correct_prefices_map = {
    "numeric_memory_f.ukbb_maximum_digits_remembered_correctly.": "numeric_memory_f_ukbb_maximum_digits_remembered_correctly.",
     "Systolic_blood_pressure.mean.": "Systolic_blood_pressure_mean.",
     "Diastolic_blood_pressure.mean.": "Diastolic_blood_pressure_mean."
}

all_cols = ["f.eid"]
for key, var_list in factors_variables_map.items():
    all_cols = all_cols + var_list

## now we prepare all the taba by merging all tables in one
df = None

for key, path_to_file in input_files.items():

    print(key)

    df_other = pd.read_table(path_to_file, nrows=n_samples, sep=seps_input_files[key])



    columns_ = df_other.columns

    # field naming corrections
    if key in ["food", "alcohol", "sociodem"]:
        for col in columns_:
            for i in range(4):
                if col.endswith("_"+str(i)):
                    ll = len(col) - 2
                    prefix = col[0:ll]
                    new_col = prefix + "."+str(i)
                    df_other = df_other.rename(columns={col: new_col})

    # another field naming correction
    if key in ["habits"]:
        df_other = df_other.rename(columns={"age_years_int": "age_years_at.0"})
    if key in ["medications"]:
        df_other = df_other.rename(columns={"aspirne_use.0": "aspirine_use.0",
                                            "aspirne_use.1": "aspirine_use.1",
                                            "aspirne_use.2": "aspirine_use.2",
                                            "aspirne_use.3": "aspirine_use.3"})

    # another field naming corrections
    if key in ["risk_factors", "cognitive"]:
        for col in columns_:
            for old_pr, new_pr in correct_prefices_map.items():
                for i in range(4):
                    if col == old_pr + str(i) + ".0":
                        new_col = new_pr + str(i)
                        df_other = df_other.rename(columns={col: new_col})
                    else:
                        if col == old_pr + str(i):
                            new_col = new_pr + str(i)
                            df_other = df_other.rename(columns={col: new_col})

    columns_to_keep = [col for col in df_other.columns if col in all_cols]
    df_other = df_other[columns_to_keep]

    if all_defined == 1.0:
        df_other = df_other.dropna()

    if key in ["diagnoses"]:
        for col in columns_to_keep:
            if col == "f.eid":
                continue
            df_other.loc[df_other[col].eq(True), col] = 1.0
            df_other.loc[df_other[col].eq(False), col] = 0.0

    print(f"have read the file for {key}")
    print(f"number of rows {df_other.shape[0]}")

    # df.set_index("f.eid")
    # df_other.set_index("f.eid")
    if df is not None:
        df = pd.merge(df, df_other, on=["f.eid"])
        print(f"number of rows after merge with main {df.shape[0]}")
    else:
        df = df_other
    print("merged the new block of data")



print(f"Number or rows: {df.shape[0]}")

## now we make a factor table
df_factors = pd.DataFrame({'Variable': pd.Series(dtype='str'),
                   'Factor': pd.Series(dtype='str'),
                   'Loading': pd.Series(dtype='float')})

if one_to_one == 1:
    for factor, var_list in factors_variables_map.items():
        for var in var_list:
            # print(var)
            list_row = [var, var, 1.0]
            df_factors.loc[len(df_factors)] = list_row
else:
    for factor, var_list in factors_variables_map.items():
        print("***")
        print(factor)
        for var in var_list:
            # print(var)
            list_row = [var, factor, 1.0]
            df_factors.loc[len(df_factors)] = list_row

df.loc[df['gender'].eq("female"), 'gender'] = 1.0
df.loc[df['gender'].eq("male"), 'gender'] = 0.0

df.to_csv(output_data_file_name, sep=',', index=False)
df_factors.to_csv(output_factor_file_name, sep=',', index=False)


