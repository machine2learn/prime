import pandas as pd
import numpy as np
from datetime import datetime

field_id = "f.eid"

field_snap_game_rt = "f.404"

diab_limit_age = 21.0
fld_age_0 = "f.21022.0.0"

comorbidities = ["Diabetes_2", "Hypertension", "Dyslipidemia", "Depression", "Diabetes_u"]

codes = {
    "20002": { # priority to the data input by the personal
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
            "E780": "Dyslipidemia",  # pure hypercholesterolaemia
            "E781": "Dyslipidemia",  # pure hyperglyceridemia
            # "E782": "Dyslipidemia", # mixed hypercholesterolaemia
            "E784": "Dyslipidemia",  # other hyperlipidaemia
            "E785": "Dyslipidemia",  # hyperlipidaemia unspecified
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
        "date": "41280",
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
            "2720": "Dyslipidemia",  # pure hypercholesterolaemia
            # "27200": "Dyslipidemia", familial
            "27201": "Dyslipidemia",  # hyper-beta lipoproteilaemia
            "27202": "Dyslipidemia",  # pure hypercholesterolaemia (group a)
            "27203": "Dyslipidemia",  # ldl  hypercholesterolaemia
            "27209": "Dyslipidemia",  # pure hypercholesterolaemia (other)
            "2721": "Dyslipidemia",  # pure hyperglyceridemia
            # "2722": "Dyslipidemia",  # mixed hyperlipidemia
            "2724": "Dyslipidemia",  # other and unsepcified hypercholesterolaemia
            "27240": "Dyslipidemia",  # hyperlipidemia of nephrotic syndrome
            "27248": "Dyslipidemia",  # hyperlipidemia of other specified courses
            "27249": "Dyslipidemia",  # hyperlipidemia not otherwise specified
            "311": "Depression",
            "319": "Depression"
        },
        "method_timing": None,
        "age": "",
        "date": "41281",
        "year": "",
        "instancing": [0],
        "array_length": 47,
        "check_timing": 1
    }

}

# n_samples = 298140 # limit #samples for debug
n_samples = None

withdrawn_participants_file = '/projects/prime/ukbb/w62644_20220222.csv'
data_filename = '/projects/prime/ukbb/04_01_2023_selected.tab'

withdrawn = pd.read_csv(withdrawn_participants_file, header=None)
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



print("start reading the table .... ")
df = pd.read_table(data_filename, sep='\t', usecols=all_fields_, dtype=str, nrows=n_samples)
# print(df['f.41281.0.46'])
# exit()
print("finished reading the table .... ")
print("# participants before checking entrance age")
print(len(df))
df_ = df[~df[fld_age_0].isnull()]
df = df_
print("# participants after checking entrance age but yeat with withdrawn ")
print(len(df))
df = df[~df[field_id].isin(withdrawn_col)]
print("# participants without withdrawn ")
print(len(df))

date_fields = ["f.53.0.0", "f.53.1.0", "f.53.2.0", "f.53.3.0"]

counters = {
    "Diabetes_2": [0, 0, 0, 0],
    "Hypertension": [0, 0, 0, 0],
    "Dyslipidemia": [0, 0, 0, 0],
    "Depression": [0, 0, 0, 0],
    "Diabetes_u": [0, 0, 0, 0]
}

diagnoses_ids_0 = {
    "Diabetes_2": [],
    "Hypertension": [],
    "Dyslipidemia": [],
    "Depression": [],
    "Diabetes_u": []
}

def diabetes_u_2():

    if (diag_age_fld in df.columns) and (codes[comor_field_ukbb]["age"] is not ""):

        if isinstance(row[diag_age_fld], str):
            diag_age = float(row[diag_age_fld])
            if diag_age >= diab_limit_age: # if the value of this field in 2976 is -1, the condition will not hold
                return True
            else:
                return False


    if (diag_date_fld in df.columns) and (codes[comor_field_ukbb]["date"] is not ""):
        if isinstance(row[diag_date_fld], str):
            if row[diag_date_fld] != "-1" and row[diag_date_fld] != "-3":
                diag_date = datetime.strptime(row[diag_date_fld], '%Y-%m-%d')
                diag_year = diag_date.year
                if birth_year + diab_limit_age <= diag_year:
                    return True
                else:
                    return False


    if (diag_year_fld in df.columns) and (codes[comor_field_ukbb]["year"] is not ""):
        if isinstance(row[diag_year_fld], str):
            diag_year = float(row[diag_year_fld])
            if diag_year >= 0:
                if birth_year + diab_limit_age <= diag_year:
                    return True
                else:
                    return False

    return None

def diab_u_2_wrapper(instance_int, ids):

    check = diabetes_u_2()
    if check is not None:
        if check:
            if found["Diabetes_2"][instance_int] is None:
                found["Diabetes_2"][instance_int] = True
                counters["Diabetes_2"][instance_int] = counters["Diabetes_2"][instance_int] + 1
                if instance_int == 0:
                    ids["Diabetes_2"] = ids["Diabetes_2"] + [(row["f.eid"])]
        else:
            found["Diabetes_2"][instance_int] = False

    return ids


def diab_u_2_wrapper_hospital(instance_int, ids):

    check = diabetes_u_2()
    if check is not None:
        if check:
            # can be false because ppl reported no diabetes during the visits
            if found["Diabetes_2"][instance_int] != True :
                found["Diabetes_2"][instance_int] = True
                counters["Diabetes_2"][instance_int] = counters["Diabetes_2"][instance_int] + 1
                if instance_int == 0:
                    ids["Diabetes_2"] =ids["Diabetes_2"] + [(row["f.eid"])]
        else:
            found["Diabetes_2"][instance_int] = False

    return ids



# loop over participants
c_p = 0


for row_index, row in df.iterrows():

    print(str(c_p))
    c_p = c_p + 1

    found = {
        "Diabetes_2": [None, None, None, None],
        "Hypertension": [None, None, None, None],
        "Dyslipidemia":  [None, None, None, None],
        "Depression":  [None, None, None, None],
        "Diabetes_u":  [None, None, None, None]
    }

    date_visit = []
    year_visit = []
    age_visit = []

    for i in range(4):
        date_str = row[date_fields[i]]
        if not isinstance(date_str, str):
            current_date = None
            current_year_visit = None
        else:
            current_date = datetime.strptime(date_str, '%Y-%m-%d')
            current_year_visit = current_date.year

        date_visit.append(current_date)
        year_visit.append(current_year_visit)
        if i == 0:
            current_age = int(row[fld_age_0])
        else:
            if not (current_year_visit is None):
                current_age = age_visit[0] + (current_year_visit - year_visit[0])
            else:
                current_age = None

        age_visit.append(current_age)

    birth_year = year_visit[0] - age_visit[0]

    for comor_col in comorbidities_fileds:

        comor_col_split = comor_col.split(".")
        # comor_col_split[0] is 'f'
        # comor_col_split[1] is ukbb field
        # comor_col_split[2] is instance between 0 and 3 (can be 0, or 0,1,2 as well)
        # comor_col_split[3] is array index in the lies for multivalued fields

        comor_field_ukbb = comor_col_split[1]
        instance = int(comor_col_split[2])
        field_codes = codes[comor_field_ukbb]["values"]

        current_val = row[comor_col]
        if current_val in field_codes:
            comor = codes[comor_field_ukbb]["values"][current_val]

            # we will have to find the instance number where this diagnosis may fit by timing
            diag_date_fld = "f." + codes[comor_field_ukbb]["date"] + "." + comor_col_split[2]
            diag_year_fld = "f." + codes[comor_field_ukbb]["year"] + "." + comor_col_split[2]
            diag_age_fld = "f." + codes[comor_field_ukbb]["age"] + "." + comor_col_split[2]
            if len(comor_col_split) == 4:
                diag_date_fld = diag_date_fld + "." + comor_col_split[3]
                diag_year_fld = diag_year_fld + "." + comor_col_split[3]
                diag_age_fld = diag_age_fld + "." + comor_col_split[3]


            if codes[comor_field_ukbb]["check_timing"]: # only for visit-less fields
                # for them instance is always zero  and the date should be filled in
                # we find all the instances after the hospitalisation date
                for i in range(4):
                    if found[comor][i] != True:
                        visit_date = date_visit[i]
                        print(diag_date_fld)
                        print(row[diag_date_fld])
                        if (not diag_date_fld.startswith("f..")) and (visit_date is not None):
                            if isinstance(row[diag_date_fld], str):
                                if row[diag_date_fld] != "-1" and row[diag_date_fld] != "-3":
                                    diag_date = datetime.strptime(row[diag_date_fld], '%Y-%m-%d')
                                    # this is actualy y hospitaisation date in general
                                    # it is considered only if the diagnos is not set up or rejected for sure (diab 2)
                                    # during the instance fields control
                                    if diag_date <= visit_date:

                                        # 1) hospitalisation date is not a diagnosis date if diab_u is not for the first time
                                        # but in this branch we see diabetes u for the first time
                                        # since the visit reports have been already checked and not detected there
                                        # 2)  there are ppl who reported no diabetes but they did have diabetes
                                        # during prior hospitalisation
                                        if comor == "Diabetes_u":
                                            diagnoses_ids_0 = diab_u_2_wrapper_hospital(i, diagnoses_ids_0)


                                        found[comor][i] = True
                                        counters[comor][i] = counters[comor][i] + 1
                                        if i == 0:
                                            diagnoses_ids_0[comor] = diagnoses_ids_0[comor] + [(row["f.eid"])]



            else:
                # here the dates, ages and years indeed dates, ages and years indeed of the first diagnostics
                if found[comor][instance] != True:
                    found[comor][instance] = True
                    counters[comor][instance] = counters[comor][instance] + 1

                    if instance == 0:
                        diagnoses_ids_0[comor] = diagnoses_ids_0[comor] + [(row["f.eid"])]

                    # in this branch we have diagnoses dates for diab_u so diab_2 can be safely updated
                    if comor == "Diabetes_u":
                        diagnoses_ids_0 = diab_u_2_wrapper(instance, diagnoses_ids_0)

##############
df_diab2_ids = pd.DataFrame()
df_diab2_ids['f.eid'] = pd.Series(np.unique(np.array(diagnoses_ids_0["Diabetes_2"])))
print(f"Ppl diagnosed diabetes-2 in visit {0}: {len(df_diab2_ids.axes[0])}")
df_diab2_ids.to_csv('/projects/prime/ukbb/python_test_diabetes_2_baseline_ids.csv', sep=';')
##############
df_hyper_ids = pd.DataFrame()
df_hyper_ids['f.eid'] = pd.Series(np.unique(np.array(diagnoses_ids_0["Hypertension"])))
print(f"Ppl diagnosed hypertension in visit {0}: {len(df_hyper_ids.axes[0])}")
df_hyper_ids.to_csv('/projects/prime/ukbb/python_test_hypertension_baseline_ids.csv', sep=';')
###########
df_fat_ids = pd.DataFrame()
df_fat_ids['f.eid'] = pd.Series(np.unique(np.array(diagnoses_ids_0["Dyslipidemia"])))
print(f"Ppl diagnosed Dyslipidemia in visit {0}: {len(df_fat_ids.axes[0])}")
df_fat_ids.to_csv('/projects/prime/ukbb/python_test_Dyslipidemia_baseline_ids.csv', sep=';')
##############
df_depr_ids = pd.DataFrame()
df_depr_ids['f.eid'] = pd.Series(np.unique(np.array(diagnoses_ids_0["Depression"])))
print(f"Ppl diagnosed Depression in visit {0}: {len(df_depr_ids.axes[0])}")
df_depr_ids.to_csv('/projects/prime/ukbb/python_test_depression_baseline_ids.csv', sep=';')
##############
df_diabU_ids = pd.DataFrame()
df_diabU_ids['f.eid'] = pd.Series(np.unique(np.array(diagnoses_ids_0["Diabetes_u"])))
print(f"Ppl diagnosed Diabetes_u in visit {0}: {len(df_diabU_ids.axes[0])}")
df_diabU_ids.to_csv('/projects/prime/ukbb/python_test_diabetes_U_baseline_ids.csv', sep=';')

