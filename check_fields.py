import pandas as pd

needed_fields_file = '/projects/prime/ukbb/05_01_2023.txt'
data_filename = '/projects/prime/ukbb/04_01_2023_selected.tab'
# data_filename = '/projects/prime/ukbb/13_10_2022_selected.tab'
fields = pd.read_csv(needed_fields_file, header=None)
fields_col = list(fields.iloc[:, 0])

df_checker = pd.read_table(data_filename, dtype=str,  sep='\t', nrows=100)


actual_cols = df_checker.head()
absent_fields = []

for field in fields_col:
    prefix = "f." + str(field) + "."
    help_closure = filter(lambda x: x.startswith(prefix), actual_cols)
    hit_list = list(help_closure)
    if len(hit_list) == 0:
        print(prefix)
        absent_fields.append(field)


### previous ######

data_filename_prev = '/projects/prime/ukbb/13_10_2022_selected.tab'
# data_filename_prev = '/projects/prime/ukbb/20220622_selected.tab'
df_checker_prev = pd.read_table(data_filename_prev, nrows=100, dtype=str, sep='\t')

prev_cols = df_checker_prev.head()
prev_absent_fields = []

print("Fields absent in the previous pull:")

for field in fields_col:
    prefix = "f." + str(field) + "."
    help_closure = filter(lambda x: x.startswith(prefix), prev_cols)
    hit_list = list(help_closure)
    if len(hit_list) == 0:
        print(prefix)
        prev_absent_fields.append(field)

