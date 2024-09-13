import pandas as pd
import numpy as np
from datetime import datetime

field_id = "f.eid"

field_snap_game_rt = "f.404"

diab_limit_age = 30.0
fld_age_0 = "f.21022.0.0"

comorbidities = ["Diabetes_2", "Hypertension", "Dyslipidemia", "Depression", "Diabetes_u"]

codes = {
    "20002": {  # priority to the data input by the personal
        "values": {
            "1223": "Diabetes_2",
            "1220": "Diabetes_u",
            "1065": "Hypertension",
            "1072": "Hypertension",
            "1473": "Dyslipidemia",
            "1286": "Depression"
        },
        "method_timing": "20013",
        "age": "20009",
        "date": "",
        "year": "20008",
        "instancing": [0, 1, 2, 3],
        "array_length": 34,
        "check_timing": 0
    },

    "6150": {
        "values": {
            "4": "Hypertension"},
        "method_timing": None,
        "age": "",
        "date": "",
        "year": "",
        "instancing": [0, 1, 2],
        "array_length": 4,
        "check_timing": 0
    },

    "2443": { # less priority because the input by a participant
        "values": {
            "1": "Diabetes_u"
        },
        "method_timing": None,
        "age": "2976",
        "date": "",
        "year": "",
        "instancing": [0, 1, 2, 3],
        "array_length": 1,
        "check_timing": 0
    },

    "41270": {
        "values": {
            "E110": "Diabetes_2",
            "E111": "Diabetes_2",
            "E112": "Diabetes_2",
            "E113": "Diabetes_2",
            "E114": "Diabetes_2",
            "E115": "Diabetes_2",
            "E116": "Diabetes_2",
            "E117": "Diabetes_2",
            "E118": "Diabetes_2",
            "E119": "Diabetes_2",
            "E140": "Diabetes_u",
            "E141": "Diabetes_u",
            "E142": "Diabetes_u",
            "E143": "Diabetes_u",
            "E144": "Diabetes_u",
            "E145": "Diabetes_u",
            "E146": "Diabetes_u",
            "E147": "Diabetes_u",
            "E148": "Diabetes_u",
            "E149": "Diabetes_u",
            "I10": "Hypertension",
            "I11": "Hypertension",
            "I110": "Hypertension",
            "I119": "Hypertension",
            "I12": "Hypertension",
            "I120": "Hypertension",
            "I129": "Hypertension",
            "I13": "Hypertension",
            "I130": "Hypertension",
            "I131": "Hypertension",
            "I132": "Hypertension",
            "I139": "Hypertension",
            "E780": "Dyslipidemia", # pure hypercholesterolaemia
            "E781": "Dyslipidemia", # pure hyperglyceridemia
            # "E782": "Dyslipidemia", # mixed hypercholesterolaemia
            "E784": "Dyslipidemia", # other hyperlipidaemia
            "E785": "Dyslipidemia", # hyperlipidaemia unspecified
            # "E789": "Dyslipidemia", # disorder of lipoproteine metabolism unspecified
            "F32": "Depression",
            "F320": "Depression",
            "F321": "Depression",
            "F322": "Depression",
            "F323": "Depression",
            "F328": "Depression",
            "F329": "Depression",
            "F33": "Depression",
            "F330": "Depression",
            "F331": "Depression",
            "F332": "Depression",
            "F333": "Depression",
            "F334": "Depression",
            "F338": "Depression",
            "F339": "Depression"},
        "method_timing": None,
        "age": "",
        "date": "41280", # is not the date by the first diagnostics per se! only hospitalisation; use if  not set by other fields
        "year": "",
        "instancing": [0],
        "array_length": 243,
        "check_timing": 1
    },
    "41271": {
        "values": {
            "25000": "Diabetes_2",
            "25009": "Diabetes_u",
            "25010": "Diabetes_2",
            "25019": "Diabetes_u",
            "250": "Diabetes_u",
            "25029": "Diabetes_u",
            "25099": "Diabetes_u",
            "25003": "Diabetes_u",
            "25004": "Diabetes_u",
            "25005": "Diabetes_u",
            "401": "Hypertension",
            "4010": "Hypertension",
            "4011": "Hypertension",
            "4019": "Hypertension",
            "402": "Hypertension",
            "4020": "Hypertension",
            "4021": "Hypertension",
            "4029": "Hypertension",
            "403": "Hypertension",
            "4030": "Hypertension",
            "4031": "Hypertension",
            "4039": "Hypertension",
            "404": "Hypertension",
            "4040": "Hypertension",
            "4041": "Hypertension",
            "4049": "Hypertension",
            "2720": "Dyslipidemia", # pure hypercholesterolaemia
            # "27200": "Dyslipidemia", familial
            "27201": "Dyslipidemia", # hyper-beta lipoproteilaemia
            "27202": "Dyslipidemia", # pure hypercholesterolaemia (group a)
            "27203": "Dyslipidemia", # ldl  hypercholesterolaemia
            "27209": "Dyslipidemia", # pure hypercholesterolaemia (other)
            "2721": "Dyslipidemia", # pure hyperglyceridemia
            # "2722": "Dyslipidemia",  # mixed hyperlipidemia
            "2724": "Dyslipidemia", # other and unsepcified hypercholesterolaemia
            "27240": "Dyslipidemia",  # hyperlipidemia of nephrotic syndrome
            "27248": "Dyslipidemia",  # hyperlipidemia of other specified courses
            "27249": "Dyslipidemia",  # hyperlipidemia not otherwise specified
            "311": "Depression",
            "319": "Depression"
        },
        "method_timing": None,
        "age": "",
        "date": "41281", # is not the date by the first diagnostics per se! only hospitalisation; use if  not set by other fields
        "year": "",
        "instancing": [0],
        "array_length": 47,
        "check_timing": 1
    },



}

date_fields = ["f.53.0.0", "f.53.1.0", "f.53.2.0", "f.53.3.0"]

def code_comorb(x, code_map, inst, fld): # not definitive yes or no, but just for this filed if there is a code or not
    if x in code_map["values"]:
        return fld == (code_map["values"][x] + "." + inst)
    else:
        return False


## in 4 methods below we do not update already set up to True values:
##  df_in[d_field].ne(True)


def diabetes_u_2_date_hospit(instance_int, comor_field_ukbb, df_in, found_como, d_field, du_field, diag_date_fld,
                             cond_date_present):
    # the hospitalisation date is not the same as the date when the desease is diagnosed
    # if the hospitalisation happened after the limit age for diabetes 2 we cannot condlue that this is not diabetes
    if codes[comor_field_ukbb]["date"] is not "":
        test = df_in["diab_limit_year"].le(pd.DatetimeIndex(df_in[diag_date_fld]).year)
        df_in.loc[found_como.eq(1.0) & df_in[d_field].ne(True) & test.eq(True), d_field] = True

    return df_in

def diabetes_u_2_date_diag(instance_int, comor_field_ukbb, df_in, found_como, d_field,  diag_date_fld, cond_date_present):

    if codes[comor_field_ukbb]["date"] is not "":
        df_in.loc[found_como.eq(1.0) & df_in[d_field].ne(True) & ~df_in[diag_date_fld].isnull() &
                  cond_date_present[instance_int], d_field] = \
            (df_in["diab_limit_year"].le(pd.DatetimeIndex(df_in[diag_date_fld]).year))

    return df_in

def diabetes_u_2_age_diag(instance_int, comor_field_ukbb, df_in,found_como, d_field,
                 diag_age_fld, cond_age_present):

    if codes[comor_field_ukbb]["age"] is not "":
        df_in.loc[found_como.eq(1.0) & df_in[d_field].ne(True) & ~df_in[diag_age_fld].isnull() &
                  cond_age_present[instance_int],  d_field] = \
            df_in[diag_age_fld].astype(float).ge(df_in["limit_diab_age"])
    return df_in


def diabetes_u_2_year_diag(instance_int, comor_field_ukbb, df_in, found_como, d_field,
                     diag_year_fld, cond_year_present):
    if codes[comor_field_ukbb]["year"] is not "":
        df_in.loc[found_como.eq(1.0) & df_in[d_field].ne(True) & ~df_in[diag_year_fld].isnull() &
                  cond_year_present[instance_int], d_field] = \
            (df_in["diab_limit_year"].le(df_in[diag_year_fld].astype(float)))

    return df_in


def preprocess(df_in):
    # preparations
    df_in.insert(1, "limit_diab_age", pd.Series(diab_limit_age, index=df_in.index))

    out_comorb_fileds = []
    cond_date_present = [None, None, None, None]
    cond_age_present = [None, None, None, None]
    cond_year_present = [None, None, None, None]

    # add some convenience columns (dates, years, ages) and  final diagnoses fields
    for i in range(4):

        # print(date_fields[i])
        current_year = pd.DatetimeIndex(df_in[date_fields[i]]).year
        df_in.insert(1, "date_visit."+str(i), df_in[date_fields[i]])
        df_in.insert(1, "year_visit."+str(i), current_year)
        if i == 0:
            current_age = df_in[fld_age_0].astype(int)
        else:
            current_age = df_in["age_visit.0"] + (current_year - df_in["year_visit.0"])

        df_in.insert(1, "age_visit."+str(i), current_age)

        # making main output columns listing comobidities
        for comorb in comorbidities:
            clmn = comorb + '.' + str(i)
            out_comorb_fileds.append(clmn)
            df_in.insert(1, clmn, pd.Series(None, index=df_in.index))

        fld_1 = "date_visit." + str(i)
        cond_date_present[i] = ~df_in[fld_1].isnull() & df_in[fld_1].ne("-1") & df_in[fld_1].ne("-3")

        fld_2 = "age_visit." + str(i)
        cond_age_present[i] = ~df_in[fld_2].isnull()
        cond_age_present[i] = cond_age_present[i] & df_in[fld_2].astype(float).ge(0)

        fld_3 = "year_visit." + str(i)
        cond_year_present[i] = ~df_in[fld_3].isnull() & df_in[fld_3].astype(float).ge(0)


    ## end convenience columns
    birth_year = df_in["year_visit.0"] - df_in["age_visit.0"]
    df_in.insert(1, "birth_year." + str(i), birth_year)
    diab_limit_year = birth_year + diab_limit_age

    df_in.insert(1, "diab_limit_year", diab_limit_year)

    # loop over the ukbb fields where comorbidities are encoded
    for comor_col in comorbidities_fileds:

        print(comor_col)

        if df_in[comor_col].isnull().all():
            continue

        comor_col_split = comor_col.split(".")
        # comor_col_split[0] is 'f'
        # comor_col_split[1] is ukbb field
        # comor_col_split[2] is instance between 0 and 3 (can be 0, or 0,1,2 as well)
        # comor_col_split[3] is array index in the lies for multivalued fields

        comor_field_ukbb = comor_col_split[1]
        instance = comor_col_split[2]

        # we will have to find the instance number where this diagnosis may fit by timing
        diag_date_fld = "f." + codes[comor_field_ukbb]["date"] + "." + comor_col_split[2]
        diag_year_fld = "f." + codes[comor_field_ukbb]["year"] + "." + comor_col_split[2]
        diag_age_fld = "f." + codes[comor_field_ukbb]["age"] + "." + comor_col_split[2]
        if len(comor_col_split) == 4:
            diag_date_fld = diag_date_fld + "." + comor_col_split[3]
            diag_year_fld = diag_year_fld + "." + comor_col_split[3]
            diag_age_fld = diag_age_fld + "." + comor_col_split[3]


        # df_in.loc[df_in[comorbidity_column].isin(code_values), out_comorbidity_column]

        if codes[comor_field_ukbb]["check_timing"] == 1: # no instancing for this ukbb field

            # loop of the summary output comorbidity columns
            for fld_out in out_comorb_fileds:
                fld_out_split = fld_out.split(".")

                i_str = fld_out_split[1]  # instance

                # check if the current comoridity fld_out is encoded by the current ukbb field value
                # and the out current comorbidity instance coinsides with the ukbb field instance index

                found_comorbidity = df_in[comor_col].map(lambda x: code_comorb(x, codes[comor_field_ukbb], i_str,
                                                                               fld_out))

                if not found_comorbidity.any():
                    continue

                fld_1 = "date_visit." + i_str

                # this is actually not a diagnose date, this is a hopitalisation date
                timing = ~df_in[diag_date_fld].isnull() & cond_date_present[int(i_str)] & \
                         (pd.to_datetime(df_in[diag_date_fld], format='%Y-%m-%d') <=
                          pd.to_datetime(df_in[fld_1], format='%Y-%m-%d'))


                # we do not update the field if it has been set to True earlier
                df_in.loc[found_comorbidity & df_in[fld_out].ne(True) & timing, fld_out] = True

                # not instanced fields does have date of hospitalisation but not necessary when diagnose has
                # not necessary diagnoses; diagnoses if for the first time

                if fld_out == ("Diabetes_u." + i_str):
                    d_field = "Diabetes_2." + i_str
                    du_field = "Diabetes_u." + i_str
                    found_comorbidity_du = df_in[fld_out]

                    df_in = diabetes_u_2_date_hospit(int(i_str), comor_field_ukbb, df_in, found_comorbidity_du,
                                                   d_field, du_field, diag_date_fld, cond_date_present)






        else:
            # loop of the summary output comorbidity columns
            for fld_out in out_comorb_fileds:

                if not fld_out.endswith(instance):
                    continue

                # check if the current output comorbidity fld_out is encoded by the current ukbb field value
                # and the current output comorbidity instance coincides with the ukbb field instance index
                found_comorbidity = df_in[comor_col].map(lambda x: code_comorb(x, codes[comor_field_ukbb], instance,
                                                                               fld_out))

                if not found_comorbidity.any():
                    continue

                df_in.loc[found_comorbidity, fld_out] = True

                # here update is safe because the reported dates are the dates of the diagnoses
                if fld_out == ("Diabetes_u." + instance):
                    d_field = "Diabetes_2." + instance
                    df_in = diabetes_u_2_age_diag(int(instance), comor_field_ukbb, df_in, found_comorbidity,
                                                   d_field, diag_age_fld, cond_age_present)
                    df_in = diabetes_u_2_date_diag(int(instance), comor_field_ukbb, df_in, found_comorbidity,
                                                   d_field, diag_date_fld, cond_date_present)
                    df_in = diabetes_u_2_year_diag(int(instance), comor_field_ukbb, df_in, found_comorbidity,
                                                  d_field, diag_year_fld, cond_year_present)


    return df_in


if __name__ == '__main__':

    n_samples = None # limit #samples for debug
    output_dir = "/projects/prime/ukbb/preprocessed_data_2023"
    out_file = "/diagnoses_2023_diabetes_30_dec_23.csv"
    data_filename = '/projects/prime/ukbb/04_01_2023_selected.tab'

    withdrawn_participants_file = '/projects/prime/ukbb/withdraw6244_207_20231205.txt'
    withdrawn = pd.read_csv(withdrawn_participants_file, header=None, dtype=str)
    withdrawn_col = list(withdrawn.iloc[:, 0])

    ### find out the list of all columns
    data_columns = pd.read_table(data_filename, nrows=1, sep='\t').columns

    fields_visit_dates = [col for col in data_columns if col.startswith("f.53.")]
    # fields_visit_ages = [col for col in data_columns if col.startswith("f.21003.")]

    all_fields_ = [field_id] + [fld_age_0] + fields_visit_dates

    print(all_fields_)

    comorbidities_fileds = []

    for key in codes:

        print(key)

        current_list = [col for col in data_columns if col.startswith("f." + key)]
        all_fields_ = all_fields_ + current_list
        comorbidities_fileds = comorbidities_fileds + current_list
        print(current_list)

        if codes[key]["method_timing"] is not None:
            prefix = "f." + codes[key]["method_timing"] + "."
            current_list = [col for col in data_columns if col.startswith(prefix)]
            all_fields_ = all_fields_ + current_list
            print(current_list)
        if codes[key]["age"] is not "":
            prefix = "f." + codes[key]["age"] + "."
            current_list = [col for col in data_columns if col.startswith(prefix)]
            all_fields_ = all_fields_ + current_list
            print(current_list)
        if codes[key]["date"] is not "":
            prefix = "f." + codes[key]["date"] + "."
            current_list = [col for col in data_columns if col.startswith(prefix)]
            all_fields_ = all_fields_ + current_list
            print(current_list)
        if codes[key]["year"] is not "":
            prefix = "f." + codes[key]["year"] + "."
            current_list = [col for col in data_columns if col.startswith(prefix)]
            all_fields_ = all_fields_ + current_list
            print(current_list)

    ###########################################
    # print(all_fields_)

    print("start reading the table .... ")
    df = pd.read_table(data_filename, sep='\t', usecols=all_fields_, dtype=str, nrows=n_samples)


    # print(df['f.41281.0.46'])
    # exit()

    print("finished reading the table .... ")
    print("# participants before checking entrance age")
    print(len(df))
    df_ = df[~df[fld_age_0].isnull()]
    df = df_
    print("# participants after checking entrance age but yet with withdrawn ")
    print(len(df))
    df = df[~df[field_id].isin(withdrawn_col)]
    print("# participants without withdrawn ")
    print(len(df))

    ### new approach /test
    # 20002
    ### "1223": "Diabetes_2",
    ### "1220": "Diabetes_u",
    ### "1065": "Hypertension",
    ### "1072": "Hypertension",
    ###  "1473": "Dyslipidemia",
    ### "1286": "Depression"
    diag_20002_fields_0 = [field_id] + [col for col in data_columns if col.startswith("f.20002.0")]
    diag_20002_dates_0 = [field_id] + [col for col in data_columns if (col.startswith("f.20008.0") |
                                                                       col.startswith("f.20009.0"))]
    diag_20002 = df[diag_20002_fields_0]
    diag_20002.insert(1, 'diab_2.0', pd.Series((diag_20002 == '1223').any(axis=1), index=df.index))
    diag_20002.insert(1, 'diab_u.0', pd.Series((diag_20002 == '1220').any(axis=1), index=df.index))
    diag_20002.insert(1, 'hyperten.0', pd.Series((diag_20002.isin(['1065', "1072"])).any(axis=1), index=df.index))
    diag_20002.insert(1, 'dyslipid.0', pd.Series((diag_20002 == '1473').any(axis=1), index=df.index))
    diag_20002.insert(1, 'depres.0', pd.Series((diag_20002 == '1286').any(axis=1), index=df.index))

    out_list_0 = ["f.eid", 'diab_2.0', 'diab_u.0', 'hyperten.0', 'dyslipid.0', 'depres.0']
    diag_20002 = diag_20002[out_list_0]



    # debu
    # df = df[df['f.eid'].isin(['1199653', '1236025'])]

    import time
    start = time.time()
    df_input = df.copy()
    df = preprocess(df_input)

    ## clean up 1: all comorbidities except Diabetes 2 are set to False if None
    comorbidities_others = ["Hypertension", "Dyslipidemia", "Depression", "Diabetes_u"]
    for i in range(4):
        date_fld = date_fields[i]
        for comor in comorbidities_others:
            fld = comor + "."+str(i)
            df.loc[df[date_fld].notna() & df[fld].isnull(), fld] = False

        fld_d2 = "Diabetes_2." + str(i)
        fld_du = "Diabetes_u." + str(i)

        new_fld = 'Diabetes_u_or_2.' + str(i)
        df.insert(1, new_fld, pd.Series(None, index=df.index))

        df.loc[df[fld_d2].isnull() & df[fld_du].eq(False) & df[date_fld].notna(), fld_d2] = False
        df.loc[df[date_fld].notna(), new_fld] = df[fld_du].eq(True) | df[fld_d2].eq(True)


    ## remove all participants for which we cannot establish diabetes of type 2 guven diabetes u
    ## may be because we do not have exact date


    # now we make a column for the diabetes_u who are not marked as diabeteds 2
    # and help list of columns which need to go to output
    out_cols = [field_id]

    for instance in range(4):
        out_cols = out_cols + ["date_visit."+str(instance), "age_visit."+str(instance)]

        for comor in comorbidities:
            clmn = comor + "."+str(instance)
            df[clmn] = df[clmn].astype(float)
            out_cols = out_cols + [clmn]

    out_cols = out_cols + ['Diabetes_u_or_2.0', 'Diabetes_u_or_2.1', 'Diabetes_u_or_2.2', 'Diabetes_u_or_2.3']

    end = time.time()
    print(f'Timing: {(end - start):.2f}s')

    count_d2_0 = df['Diabetes_2.0'].eq(1).sum()
    print(f"Ppl diagnosed diabetes 2 in visit 0 {count_d2_0}")

    ### check udenfined ###
    count_d2_0_undefined = df['Diabetes_2.0'].isnull().sum()
    print(f"checkup  NaN diagnosed diabetes 2 in visit 0 {count_d2_0_undefined}")

    check_count_d2_0 = diag_20002['diab_2.0'].sum()
    print(f"Check Ppl diagnosed diabetes 2 in visit 0, only 20002 field {check_count_d2_0}")

    # debug info
    set_main_2 = set(df[df['Diabetes_2.0'].eq(1)]['f.eid'])
    # print(set_main)
    set_check_2 = set(diag_20002[diag_20002['diab_2.0'].eq(1)]['f.eid'])
    # print(set_check)

    missing_2 = list(set_check_2 - set_main_2)

    print("missing 2 check:")
    print(missing_2)

    df_missing_2 = pd.DataFrame(data={"id_missing_2": missing_2})
    df_missing_2.to_csv(output_dir + "/debug/missing_2_ids.csv")

    # print("added comp to 20002  check:")
    added_2 = list(set_main_2 - set_check_2)
    # print(added)
    df_added_2 = pd.DataFrame(data={"id_added_2": added_2})
    df_added_2.to_csv(output_dir + "/debug/added_2_ids.csv")
    # end debug info

    count_du_0 = df['Diabetes_u.0'].sum()
    print(f"Ppl diagnosed diabetes u in visit 0 {count_du_0}")


    check_count_du_0 = diag_20002['diab_u.0'].sum()
    print(f"Check Ppl diagnosed diabetes u in visit 0, only 20002 field {check_count_du_0}")

    # debug info
    set_main = set(df[df['Diabetes_u.0'].eq(1)]['f.eid'])
    # print(set_main)
    set_check =set(diag_20002[diag_20002['diab_u.0'].eq(1)]['f.eid'])
    # print(set_check)

    missing = list(set_check - set_main)

    print("missing U check:")
    print(missing)

    df_missing = pd.DataFrame(data={"id_missing": missing})
    df_missing.to_csv(output_dir + "/debug/missing_u_ids.csv")

    # print("added comp to 20002  check:")
    added = list(set_main - set_check)
    # print(added)
    df_added = pd.DataFrame(data={"id_added": added})
    df_added.to_csv(output_dir + "/debug/added_u_ids.csv")

    count_h2_0 = df['Hypertension.0'].sum()
    print(f"Ppl diagnosed hypertension in visit 0 {count_h2_0}")

    count_lipid2_0 = df['Dyslipidemia.0'].sum()
    print(f"Ppl diagnosed Dyslipidemia in visit 0 {count_lipid2_0}")

    count_depression_0 = df['Depression.0'].sum()
    print(f"Ppl diagnosed Depression in visit 0 {count_depression_0}")

    count_du_0 = df['Diabetes_u.0'].sum()
    print(f"Ppl diagnosed diabetes unspecified in visit 0 {count_du_0}")

    count_du2_0 = df['Diabetes_u_or_2.0'].sum()
    print(f"Ppl diagnosed diabetes unspecified or t2 at visit 0 {count_du2_0}")

    print("******")

    count_d2_1 = df['Diabetes_2.1'].sum()
    print(f"Ppl diagnosed diabetes in visit 1 {count_d2_1}")

    count_h2_1 = df['Hypertension.1'].sum()
    print(f"Ppl diagnosed hypertension in visit 1 {count_h2_1}")

    count_lipid2_1 = df['Dyslipidemia.1'].sum()
    print(f"Ppl diagnosed Dyslipidemia in visit 1 {count_lipid2_1}")

    count_depression_1 = df['Depression.1'].sum()
    print(f"Ppl diagnosed Depression in visit 1 {count_depression_1}")

    count_du_1 = df['Diabetes_u.1'].sum()
    print(f"Ppl diagnosed diabetes  undefined in visit 1 {count_du_1}")

    count_du2_1 = df['Diabetes_u_or_2.1'].sum()
    print(f"Ppl diagnosed diabetes unspecified or t2 in visit 1 {count_du2_1}")

    print("******")

    # there is an undefined diabetes case where we cannot establish if it is t 2 for sure (e.g. hospitalisation happened
    # after the limit year but hospitalisation is not the date when diabetes u was first established
    # uncertain to drop out

    df = df[~df["Diabetes_2.0"].isnull()]

    df_out = df[out_cols]

    df_out.to_csv(output_dir + out_file, sep=';')

    exit(0)

    # debug diabetes 2
    df_cross_check = pd.read_table(output_dir + '/debug/python_test_diabetes_2_baseline_ids.csv', dtype=str, sep=';')
    list_cross_check = np.unique(np.array(df_cross_check['f.eid'].tolist()))
    print("Debug: the amount of participants d2 at 0 in cross check")
    print(len(list_cross_check))

    df_sinners_1 = df[df["Diabetes_2.0"].ne(True) & df['f.eid'].isin(list_cross_check)]
    df_sinners_1.to_csv(output_dir + '/debug/python_preprocessed_d2_not_true.csv', sep=';')

    df_sinners_2 = df[df["Diabetes_2.0"].ne(1.0) & df['f.eid'].isin(list_cross_check)]
    df_sinners_2.to_csv(output_dir + '/debug/python_preprocessed_d2_not_1.csv', sep=';')
    df_sinners_3 = df[df["Diabetes_2.0"].isnull() & df['f.eid'].isin(list_cross_check)]
    df_sinners_3.to_csv(output_dir + '/debug/python_preprocessed_d2_isnull.csv', sep=';')

    df_sinners_4 = df[df["Diabetes_2.0"].eq(1.0) & ~df['f.eid'].isin(list_cross_check)]
    df_sinners_4.to_csv(output_dir + '/debug/python_preprocessed_d2_not_in_check.csv', sep=';')

    # debug hypertension
    df_cross_check = pd.read_table(output_dir + '/debug/python_test_hypertension_baseline_ids.csv', dtype=str,
                                   sep=';')
    list_cross_check = df_cross_check['f.eid'].tolist()

    df_sinners_1 = df[df["Hypertension.0"].ne(True) & df['f.eid'].isin(list_cross_check)]
    df_sinners_1.to_csv(output_dir + '/debug/python_preprocessed_hypertension_not_true.csv', sep=';')

    df_sinners_4 = df[df["Hypertension.0"].eq(1.0) & ~df['f.eid'].isin(list_cross_check)]
    df_sinners_4.to_csv(output_dir + '/debug/python_preprocessed_hypertension_not_in_check.csv', sep=';')

    # debug dislipidemia
    df_cross_check = pd.read_table(output_dir + '/debug/python_test_Dyslipidemia_baseline_ids.csv', dtype=str,
                                   sep=';')
    list_cross_check = df_cross_check['f.eid'].tolist()

    df_sinners_1 = df[df["Dyslipidemia.0"].ne(True) & df['f.eid'].isin(list_cross_check)]
    df_sinners_1.to_csv(output_dir + '/debug/python_preprocessed_Dyslipidemia_not_true.csv', sep=';')

    df_sinners_4 = df[df["Dyslipidemia.0"].eq(1.0) & ~df['f.eid'].isin(list_cross_check)]
    df_sinners_4.to_csv(output_dir + '/debug/python_preprocessed__Dyslipidemia_not_in_check.csv', sep=';')

    # debug depression
    df_cross_check = pd.read_table(output_dir + '/debug/python_test_depression_baseline_ids.csv', dtype=str,
                                   sep=';')
    list_cross_check = df_cross_check['f.eid'].tolist()

    df_sinners_1 = df[df["Depression.0"].ne(True) & df['f.eid'].isin(list_cross_check)]
    df_sinners_1.to_csv(output_dir + '/debug/python_preprocessed_depression_not_true.csv', sep=';')

    df_sinners_4 = df[df["Depression.0"].eq(1.0) & ~df['f.eid'].isin(list_cross_check)]
    df_sinners_4.to_csv(output_dir + '/debug/python_preprocessed_depression_not_in_check.csv', sep=';')

    # debug d u
    df_cross_check = pd.read_table(output_dir + '/debug/python_test_diabetes_U_baseline_ids.csv', dtype=str,
                                   sep=';')
    list_cross_check = df_cross_check['f.eid'].tolist()

    df_sinners_1 = df[df["Diabetes_u.0"].ne(True) & df['f.eid'].isin(list_cross_check)]
    df_sinners_1.to_csv(output_dir + '/debug/python_preprocessed_Diabetes_u_not_true.csv', sep=';')

    df_sinners_4 = df[df["Diabetes_u.0"].eq(1.0) & ~df['f.eid'].isin(list_cross_check)]
    df_sinners_4.to_csv(output_dir + '/debug/python_preprocessed_Diabetes_u_not_in_check.csv', sep=';')



