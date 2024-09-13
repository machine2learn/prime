import pandas as pd
import numpy as np


fld_age_0 = "f.21022.0.0"
field_id = "f.eid"
diab_age_limit = 30

diag_prefix = "f.20002.0."
diag_year = "f.20008.0."
diag_age = "f.20009.0."

snap_game_rt_2 = "f.20023.2.0"
hospital_prefix = "f.41270.0"

diabetes_2_hospital_41270 = ["E110", "E111",
            "E112",
            "E113",
            "E114",
            "E115",
            "E116",
            "E117",
            "E118",
            "E119"]

diabetes_u_hospital_41270 = ["E140",
            "E141",
            "E142",
            "E143",
            "E144",
            "E145",
            "E146",
            "E147",
            "E148",
            "E149"]

diabetes_2_old_hospital_41271 = ["25000", "25010"]

diabetes_u_old_hospital_41271 = [
            "25009",
            "25019",
            "250",
            "25029",
            "25099",
            "25003",
            "25004",
            "25005"]

manual_diab_u = "f.2443.0.0"
age_manual_diab_u = "f.2976.0.0"

if __name__ == '__main__':

    n_samples = None# limit #samples for debug
    data_filename = "/projects/prime/ukbb/04_01_2023_selected.tab"
    output_dir = "/projects/prime/ukbb/preprocessed_data_2023/"

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw62644_107.txt'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    diag_flds = [col for col in data_columns if col.startswith(diag_prefix)]
    diag_year_flds = [col for col in data_columns if col.startswith(diag_year)]
    diag_age_flds = [col for col in data_columns if col.startswith(diag_age)]
    visit_date_fld = [col for col in data_columns if col.startswith("f.53.0")]

    hospital_flds = [col for col in data_columns if col.startswith("f.41270.")]
    hospital_dates = [col for col in data_columns if col.startswith("f.41280.")]

    old_hospital_flds = [col for col in data_columns if col.startswith("f.41271.")]
    old_hospital_dates = [col for col in data_columns if col.startswith("f.41281.")]

    manual_flds = [col for col in data_columns if col.startswith(manual_diab_u)]
    manual_age_flds = [col for col in data_columns if col.startswith(age_manual_diab_u)]

    snap_game_flds = [col for col in data_columns if col.startswith(snap_game_rt_2)]

    flds = [field_id, fld_age_0] + diag_flds + diag_year_flds + diag_age_flds + visit_date_fld + \
           hospital_flds + hospital_dates + old_hospital_flds + old_hospital_dates + \
           manual_flds + manual_age_flds + snap_game_flds
    print("start reading the table .... ")
    df = pd.read_table(data_filename, sep='\t', usecols=flds, dtype=str, nrows=n_samples)

    print("finished reading the table .... ")
    print(f"# participants before checking entrance age {len(df)}")

    df_ = df[~df[fld_age_0].isnull()]
    df = df_
    print(f"# participants after checking entrance age but yet with withdrawn {len(df)}")

    df = df[~df[field_id].isin(withdrawn_col)]
    print(f"# participants without withdrawn: {len(df)}")

    df["diab_2_by_20002.0"] = False
    df["diab_u_by_20002.0"] = False

    df["diab_u_by_20002.0_age_fits"] = False
    df["diab_u_by_20002.0_year_fits"] = False

    visit_year = pd.DatetimeIndex(df["f.53.0.0"]).year
    birth_year = visit_year - df[fld_age_0].astype(float)

    for diag_fld in diag_flds:
        print(diag_fld)
        diag_fld_split = diag_fld.split(".")
        index = diag_fld_split[-1]

        age_fld = diag_age + index
        age_criterion = df[diag_fld].astype(float).eq(1220) & df[age_fld].astype(float).ge(diab_age_limit)

        diag_year_fld = diag_year + index
        computed_age_diag = df[diag_year_fld].astype(float) - birth_year
        computed_age_criterion = df[diag_fld].astype(float).eq(1220) & computed_age_diag.ge(diab_age_limit)

        age_criterion = age_criterion | computed_age_criterion

        df["diab_2_by_20002.0"] = df["diab_2_by_20002.0"] | df[diag_fld].astype(float).eq(1223)
        df["diab_u_by_20002.0"] = df["diab_u_by_20002.0"] | df[diag_fld].astype(float).eq(1220)
        df["diab_u_by_20002.0_age_fits"] = df["diab_u_by_20002.0_age_fits"] | age_criterion


    df["diab_2_by_20002.0_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_2_by_20002.0"]
    df["diab_u_by_20002.0_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_u_by_20002.0"]
    df["diab_u_by_20002.0_age_fits_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_u_by_20002.0_age_fits"]

    count_diab_2 = df["diab_2_by_20002.0"].sum()
    print(f"# diabetes 2 by 20002 {count_diab_2}")

    count_diab_u = df["diab_u_by_20002.0"].sum()
    print(f"# diabetes u by 20002 {count_diab_u}")

    count_diab_u_age_fits = df["diab_u_by_20002.0_age_fits"].sum()
    print(f"# diabetes u by 20002 _age_fits {count_diab_u_age_fits}")

    count_diab_2_sg = df["diab_2_by_20002.0_and_sg"].sum()
    print(f"# diabetes 2 by 20002  and second visit sg {count_diab_2_sg}")

    count_diab_u_sg = df["diab_u_by_20002.0_and_sg"].sum()
    print(f"# diabetes u by 20002  and second visit sg {count_diab_u_sg}")

    count_diab_u_age_fits_sg = df["diab_u_by_20002.0_age_fits_and_sg"].sum()
    print(f"# diabetes u by 20002 _age_fits and second visit sg {count_diab_u_age_fits_sg}")


    ###### hospital #####
    df["diab_2_by_41270"] = False
    for hospit_fld in hospital_flds:
        print(hospit_fld)
        hosp_fld_split = hospit_fld.split(".")
        index = hospit_fld[-1]

        year_fld = "f.41280.0." + index
        year_hosp = pd.DatetimeIndex(df[year_fld]).year
        debug_criterion = (year_hosp <= visit_year)
        criterion = (year_hosp <= visit_year) & df[hospit_fld].isin(diabetes_2_hospital_41270)
        df["diab_2_by_41270"] = df["diab_2_by_41270"] | criterion

        computed_age_diag = year_hosp - birth_year
        diag_u_criterion = (year_hosp <= visit_year) & df[hospit_fld].isin(diabetes_u_hospital_41270) & \
                           computed_age_diag.ge(diab_age_limit)
        df["diab_2_by_41270"] = df["diab_2_by_41270"] | diag_u_criterion

    df["diab_2_by_41270_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_2_by_41270"]
    print("hospitalisation 41270")
    count_41270_d2 = df["diab_2_by_41270"].sum()
    print(f"# diabetes hosp 41270 {count_41270_d2}")
    count_41270_d2_and_sg = df["diab_2_by_41270_and_sg"].sum()
    print(f"# diabetes hosp 41270 and second visit sg {count_41270_d2_and_sg}")

    ###### old hospital #####
    df["diab_2_by_41271"] = False
    for old_hospit_fld in old_hospital_flds:
        print(old_hospit_fld)
        old_hosp_fld_split = old_hospit_fld.split(".")
        index = old_hospit_fld[-1]

        old_year_fld = "f.41281.0." + index
        old_year_hosp = pd.DatetimeIndex(df[year_fld]).year
        old_debug_criterion = (old_year_hosp <= visit_year)
        old_criterion = (old_year_hosp <= visit_year) & df[old_hospit_fld].isin(diabetes_2_old_hospital_41271)
        df["diab_2_by_41271"] = df["diab_2_by_41271"] | old_criterion

        computed_age_diag = old_year_hosp - birth_year
        diag_u_criterion = (old_year_hosp <= visit_year) & df[hospit_fld].isin(diabetes_u_old_hospital_41271) & \
                           computed_age_diag.ge(diab_age_limit)
        df["diab_2_by_41271"] = df["diab_2_by_41271"] | diag_u_criterion

    df["diab_2_by_41271_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_2_by_41271"]
    print("hospitalisation 41271")
    count_41271_d2 = df["diab_2_by_41271"].sum()
    print(f"# diabetes hosp 41271  {count_41271_d2}")
    count_41271_d2_and_sg = df["diab_2_by_41271_and_sg"].sum()
    print(f"# diabetes hosp 41271 and second visit sg {count_41271_d2_and_sg}")

    ###### manual #####
    df["diab_2_by_2443.0.0"] = df[manual_diab_u].astype(float).eq(1) & \
                               df[age_manual_diab_u].astype(float).ge(diab_age_limit)


    df["diab_2_by_2443.0.0_and_sg"] = df[snap_game_rt_2].astype(float).ge(0) & df["diab_2_by_2443.0.0"]
    print("manual 2443.0.0")
    count_2443_d2 = df["diab_2_by_2443.0.0"].sum()
    print(f"# diabetes manual 2443.0  {count_2443_d2}")
    count_2443_d2_and_sg = df["diab_2_by_2443.0.0_and_sg"].sum()
    print(f"# diabetes manual 2443 and second visit sg {count_2443_d2_and_sg}")

    #### summary
    print("*****")
    df["all_diab_2_and_sg"] = df["diab_2_by_20002.0_and_sg"] | df["diab_u_by_20002.0_age_fits_and_sg"] | \
                              df["diab_2_by_41270_and_sg"] | df["diab_2_by_41271_and_sg"] | \
                              df["diab_2_by_2443.0.0_and_sg"]
    count_all_d2_and_sg = df["all_diab_2_and_sg"].sum()
    print(f"# all diabetes and second visit sg {count_all_d2_and_sg}")