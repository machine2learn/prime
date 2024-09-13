import pandas as pd
import numpy as np
# import gspread
from datetime import datetime

field_id = "f.eid"
diab_limit_age = 30.0
fld_age_0 = "21022.0.0"
visit_date_fields = ["53.0.0", "53.1.0", "53.2.0", "53.3.0"]

if __name__ == '__main__':

    n_samples = None  # limit #samples for debug
    output_dir = "/projects/prime/ukbb/preprocessed_data_2023"
    out_file = "/diagnoses_2024.csv"
    data_filename = '/projects/prime/ukbb/17_01_2024_selected.tab'

    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw6244_207_20231205.txt'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None, dtype=str)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    # spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1CqFRR9-AbpIncYEIDwytY32rEqwuND1DFIw4BFmSFuw/edit#gid=1498775677'
    # sh = gc.open_by_url(spreadsheet_url)
    fld_descriptor = pd.read_csv("PRIME_variables_alignment_predimed_ukbb - Diagnoses.csv", dtype='str')

    all_fields = ["f.eid"]
    code_flds = fld_descriptor["UKBB_field"].dropna().unique().tolist()
    year_diag_flds = fld_descriptor["Field_year_diagnosed"].dropna().unique().tolist()
    age_diag_flds = fld_descriptor["Field_age_diagnosed"].dropna().unique().tolist()
    date_hospital_flds = fld_descriptor["Field_date_hospital_report"].dropna().unique().tolist()

    fld_descriptor_map = dict()
    fld_code_comorbidity_map = dict()
    fld_year_map = dict()
    fld_age_map = dict()
    fld_hospital_map = dict()

    for fld in code_flds:

        fld_descriptor_map[fld] = fld_descriptor[fld_descriptor["UKBB_field"] == fld].reset_index(drop=True)
        fld_code_comorbidity_map[fld] = dict(zip(fld_descriptor_map[fld]['UKBB_code'],
                                                 fld_descriptor_map[fld]['Comorbidity']))
        tmp = fld_descriptor_map[fld]['Field_age_diagnosed']
        if tmp[0] is not None:
            fld_age_map[fld] = tmp[0]

        tmp = fld_descriptor_map[fld]['Field_year_diagnosed']
        if tmp[0] is not None:
            fld_year_map[fld] = tmp[0]

        tmp = fld_descriptor_map[fld]['Field_date_hospital_report']
        if tmp[0] is not None:
            fld_hospital_map[fld] = tmp[0]

        code_columns = [col for col in data_columns if col.startswith("f." + fld)]
        all_fields = all_fields + code_columns

    non_code_fields = year_diag_flds + age_diag_flds + date_hospital_flds + [fld_age_0] + visit_date_fields
    for fld in non_code_fields:
        print(fld)
        columns = [col for col in data_columns if col.startswith("f." + fld)]
        print(columns)
        all_fields = all_fields + columns

    print("start reading the table .... ")
    df = pd.read_table(data_filename, sep='\t', usecols=all_fields, dtype=str, nrows=n_samples)
    print("finished reading the table .... ")
    print("# participants before checking entrance age")
    print(len(df))
    df = df[~df["f."+fld_age_0].isnull()]
    print("# participants after checking entrance age but yet with withdrawn ")
    print(len(df))
    df = df[~df[field_id].isin(withdrawn_col)]
    print("# participants without withdrawn ")
    print(len(df))

    fld_df_map = dict()
    comorbidities = fld_descriptor['Comorbidity'].unique().tolist()

    baseline_year = pd.to_datetime(df["f."+visit_date_fields[0]]).dt.year
    baseline_age = df["f."+fld_age_0].astype(float)
    birth_year = baseline_year - baseline_age
    n = len(birth_year)

    # so far consider only baseline instance
    out_clmns = [field_id, "f." + fld_age_0, "f."+visit_date_fields[0]]
    margin = pd.Series([0.5] * n)

    for X in range(1):
        d2_fld = "Diabetes_2." +str(X)
        comor_instances = []

        # initialisation of output fields: comorbidities and ages/years/dates
        for comor in comorbidities:

            comor_instance = comor + "." + str(X)
            comor_age = comor + ".age." + str(X)
            comor_year = comor + ".year." + str(X)
            comor_hospital = comor + ".hospital." + str(X)

            out_clmns = out_clmns + [comor_instance, comor_age, comor_year, comor_hospital]

            df[comor_instance] = pd.Series([False] * n)
            df[comor_age] = pd.Series([np.nan] * n)
            df[comor_year] = pd.Series([np.nan] * n)
            df[comor_hospital] = pd.Series([np.nan] * n)
            print(f"initiaised: {comor_instance} and related date-age fields")

        # fld runs over 20002.X.Y etc
        for fld in code_flds:
            cols = [col for col in data_columns if col.startswith("f." + fld + "."+str(X))]
            fld_df_map[fld] = df[["f.eid"] + cols] # data subset with projected to this fields

            for clmn in cols: # run via the whole array of code-columns
                # current column df[clmn] with the codes
                print(f"UKBB code column: {clmn}")
                ## TODO: how nans and empties are mapped?
                df['test'] = df[clmn].map(fld_code_comorbidity_map[fld]) # turn the code into comorbidity term
                clmn_split = clmn.split(".")
                index = clmn_split[-1] # take the index of the array to synchronise with the age/date array

                # check which decoded values fit which comorbidities
                for comor in comorbidities:

                    comor_instance_current = comor + "." + str(X)
                    comor_age_current = comor + ".age." + str(X)
                    comor_year_current = comor + ".year." + str(X)
                    comor_hospital_current = comor + ".year." + str(X)

                    the_first_occurence = df[comor_instance_current].eq(False) & df['test'].eq(comor)

                    # update the output comorbidity column
                    df[comor_instance_current] = df[comor_instance_current] | df['test'].eq(comor)

                    # update age column
                    if pd.notna(fld_age_map[fld]):
                        comor_age_field = "f." + fld_age_map[fld] + "."+str(X) + "." + str(index)
                        # print(f"age {comor_age_field}")
                        # first occurence of the comorbidity, fix age
                        # TODO : check how .loc works with predicates
                        df[comor_age_field].astype(float)
                        df.loc[the_first_occurence, comor_age_current] = df[comor_age_field].astype(float)
                        df.loc[df[comor_age_current].lt(0), comor_age_current] = pd.Series([np.nan] * n)

                    # update year column
                    if pd.notna(fld_year_map[fld]):
                        comor_year_field = "f." + fld_year_map[fld] + "." + str(X) + "." + str(index)
                        # print(f"year {comor_year_field}")
                        # first occurence of the comorbidity, fix year
                        df.loc[the_first_occurence, comor_year_current] = df[comor_year_field].astype(float)
                        df.loc[df[comor_year_current].lt(0), comor_year_current] = pd.Series([np.nan] * n)

                    # update hospital year,year, age diagnoses and age column
                    if pd.notna(fld_hospital_map[fld]):
                        comor_hospital_field = "f." + fld_hospital_map[fld] + "." + str(X) + "." + str(index)
                        # print(f"hospital {comor_hospital_field}")
                        # first occurence of the comorbidity, fix year hopsitalisation
                        df.loc[the_first_occurence, comor_hospital_current] = \
                            pd.to_datetime(df[comor_hospital_field]).dt.year

                    # update year if hospitalisation is known but year not
                    df.loc[the_first_occurence & df[comor_hospital_current].notna() & df[comor_year_current].isna(),
                           comor_year_current] = \
                        df[comor_hospital_current]
                    # update age if year is known but age not
                    df.loc[the_first_occurence & df[comor_year_current].notna() & df[comor_age_current].isna(),
                           comor_age_current] = \
                        df[comor_year_current] - birth_year

                    # sanity: if diagnose age is greater than the first visit age + 0.5 year, set the diagnose value to false
                    df.loc[(df[comor_age_current]-baseline_age).gt(margin), comor_instance_current] = \
                        pd.Series([False] * n)

                    if comor == "Diabetes_u":
                        df.loc[df[comor_age_current].ge(diab_limit_age), d2_fld] = df[d2_fld] | \
                                                                                   df[comor_instance_current]


    for X in range(1):
        for comor in comorbidities:
            comor_instance = comor + "." + str(X)
            df.loc[pd.Series([True] * n), comor_instance] = df[comor_instance].astype(int)
            check = df[comor_instance].sum()
            print(f"detected {comor_instance} : {check} ")

    df_out = df[out_clmns]
    df_out.to_csv(output_dir + out_file, sep=',')




