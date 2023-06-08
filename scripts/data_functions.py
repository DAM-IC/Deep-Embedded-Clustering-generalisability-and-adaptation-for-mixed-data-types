# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:57:45 2022

@author: Jip de Kok

This file contains custom functions for loading and processing the data.
"""
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer # This imputation method is no longer supported
from sklearn.impute import IterativeImputer # This imputation method is no longer supported
from sklearn.decomposition import PCA

def load_mumc_data(seed = 5192, scale_data = False, sample_selection = False,
                   time_filter = None, medium_care = False):
    '''
    This functions load and prepares the data from MUMC+ for clustering
    analysis.

    Parameters
    ----------
    seed : int, optional
        integer specifying the random seed for reproducible random operations.
        The default is 5192.
    pca : boolean, optional
        Boolean indicating whether or not to perform pca. The default is False.
    scale_data : boolean, optional
        Boolean indicating whether to scale the data with a robust scaler or
        not. The default is False.
    sample_selection : boolean, optional
        Boolean indicating whether to apply additional sample filtering based
        only including vetnillated patients, and patients with NOR. The default
        is False.
    time_filter : float, optional
        A number indicating the minimum length of stay (in hours) of a patient
        to be included in the final set. The default is None.
    medium_care : boolean, optional
        Boolean indicating whether medium care patients should be included in
        the final set. The default is False.

    Returns
    -------
    x : numpy.ndarray
        The data matrix containing all samples and variables for the custering
        analysis.
    y_mort : numpy.ndarray
        Dependent variables array regarding mortality of the samples.
    y_aki : numpy.ndarray
        Dependent variables array regarding aki of the samples.
    var_arr : list
        List containing the names of all variables in x.
    descriptives : DataFrame
        Dataframe containing additional information of the samples.
    '''
    
    # Load the data sets
    df = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/Basisdataset/ICCA "\
                     "data/Cleaned/labTable.csv",
                     encoding= 'latin',
                     sep = "|",
                     dtype={'encounterId' :             np.int32,
                            'clinicalUnitId' :          str,
                            'chartTime' :               str,
                            'intervention' :            str,
                            'testState' :               str,
                            'fillerNumber' :            str,
                            'AbnormaleMarkeringen' :    str,
                            'Eenheden' :                str,
                            'LabUitslag' :              np.float64,
                            'Referentiebereik' :        str,
                            'Specimen' :                str,
                            'Status' :                  str},
                     parse_dates = ["chartTime"])
    
    df_meta = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS MUMC "\
                          "dataset/static_input_MUMC_v8.csv")
    descriptives = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS MUMC"\
                               " dataset/descriptives_MUMC_10.csv",
                               encoding= 'unicode_escape')
    death_date = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/Basisdataset"\
                             "/ICCA data/cleaned/dateOfDecease.csv")
    comorbidities = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS MUMC"\
                               " dataset/mumcVoorgeschiedenis.csv",
                               encoding= 'unicode_escape')
    
    # Ensure that df_meta and descriptives contain identical samples
    df_meta = df_meta[df_meta["ptId"].isin(descriptives["ptId"])]
    descriptives = descriptives[descriptives["ptId"].isin(df_meta['ptId'])]
    
    # Add date of death to descriptives
    descriptives = pd.merge(descriptives,
                            death_date[["encounterId", "date_of_decease"]],
                            how = "left",
                            left_on = "ptId", right_on = "encounterId")
    
    # Conver icu admission to datetime
    descriptives.icu_admission = pd.to_datetime(descriptives.icu_admission)
    
    # Only keep lab measurements from blood specimen
    df = df.loc[(df.intervention.str.rfind("[bloed]") != -1) |
                (df.Specimen == "Blood")]
    
    # Get df in the right format
    df = df[["encounterId", "intervention", "chartTime", "LabUitslag",
             "clinicalUnitId"]]
    
    # Add comorbidities to df_meta
    df_meta = pd.merge(df_meta, comorbidities, left_on = "ptId",
                       right_on = "encounterId", how = "left")
    
    # Merge the two kidney insufficiency vairables in comorbidities and drop
    # "Geen" variable
    comorbidities["Chronische nier insufficiÃ«ntie"][comorbidities[
        "Chronische nierinsufficiÃ«ntie"] == 1] = 1
    comorbidities.drop(["Chronische nierinsufficiÃ«ntie", "Geen"], axis =1,
                       inplace = True)
    
    
    # Rename the columns to match the data from sics
    df.columns = ["studyID", "value_name", "t_lab", "value", "clinicalUnitId"]
    df_meta.rename({"ptId"          : "studyID", 
                    "ptLoS"         : "LOS_hours",
                    "sex"           : "gender",
                    "SAPSII_score"  : "sapsII_score",
                    "mech_vent"     : "mech_vent_admission",
                    "Dysritmie"     : "atrial_fibrillation",
                    "AIDS"          : "vg_aids",
                    "COPD"          : "vg_copd",
                    "Diabetes"      : "vg_dm",
                    "Myocard infarct voor IC opname"    : "vg_mi",
                    "Chron. cardiovasc.insufficiÃ«ntie" : "vg_chroncvdinsuff",
                    "Chron. resp. insufficiÃ«ntie"      : "vg_respinsuff",
                    "Chronische dialyse"                : "vg_dialysis",
                    "Chronische nier insufficiÃ«ntie"   : "vg_chronnierinsuff",
                    "Cirrhose"                          : "vg_cirrhose",
                    "Gemetastaseerd neoplasma"          : "vg_malign_meta",
                    "Hematologische maligniteit"        : "vg_malign_hema",
                    "Immunologische insufficiÃ«ntie"    : "vg_immun_insuff",
                    
                    }, inplace = True, axis = 1)
    
    # Rename values to match data from sics
    df.value_name.replace({
        'ALAT [bloed]'                  : 'ALAT (BL)',
        'Albumine [bloed]'	            : 'Albumine (BL)',
        'Alkalische fosfatase [bloed]'  : 'Alkalische fosfatase (BL)',
        'Amylase [bloed]'               : 'Amylase (BL)',
        'ASAT [bloed]'                  : 'ASAT (BL)',
        'Totaal bilirubine [bloed]'     : 'Bilirubine totaal (BL)',
        'Calcium totaal [bloed]'        : 'Calcium (BL)',
        'Chloride [bloed]'              : 'Chloride (BL)',
        'CK [bloed]'                    : 'CK (BL)',
        'CRP [bloed]'                   : 'CRP (BL)',
        'Totaal eiwit [bloed]'          : 'Eiwit totaal (BL)',
        'Fibrinogeen [bloed]'           : 'Fibrinogeen (BL)',
        'Fosfaat [bloed]'               : 'Fosfaat (BL)',
        'Gamma-GT [bloed]'              : 'Gamma-GT (BL)',
        'Hemoglobine [bloed]'           : 'Hb (BL)',
        'Hematocriet [bloed]'           : 'Ht (BL)',
        'Kalium [bloed]'                : 'Kalium (BL)',
        'Kreatinine [bloed]'            : 'Kreatinine (BL)',
        'LD [bloed]'                    : 'LDH (BL)',
        'Leucocyten [bloed]'            : 'Leukocyten (BL)',
        'Magnesium [bloed]'             : 'Magnesium (BL)',
        'MCV [bloed]'                   : 'MCV (BL)',
        'Natrium [bloed]'               : 'Natrium (BL)',
        'Trombocyten [bloed]'           : 'Trombocyten (BL)',
        'HS TNT[bloed]'                 : 'Troponine T (BL)',
        'Ureum [bloed]'                 : 'Ureum (BL)',
        'Totaal eiwit [bloed]'          : 'Eiwit totaal (BL)',
        'MCV [bloed]'                   : 'MCV (BL)'
        }, inplace = True)
    
    # Remove values that are not present in the sics data.
    # We only keep measurements taken from blood specimen
    df = df[df.value_name.isin(
            ['ALAT (BL)', 'Albumine (BL)', 'Alkalische fosfatase (BL)',
             'Amylase (BL)', 'ASAT (BL)', 'Bilirubine totaal (BL)',
             'Calcium (BL)', 'Chloride (BL)', 'CK (BL)', 'CRP (BL)',
             'Eiwit totaal (BL)', 'Fibrinogeen (BL)', 'Fosfaat (BL)',
             'Gamma-GT (BL)', 'Hb (BL)', 'Ht (BL)', 'Kalium (BL)',
             'Kreatinine (BL)', 'LDH (BL)', 'Leukocyten (BL)',
             'Magnesium (BL)', 'MCV (BL)','Natrium (BL)','Trombocyten (BL)',
             'Troponine T (BL)', 'Ureum (BL)'])]
    
    # Set index to sample ID
    df_meta.set_index("studyID", inplace = True)
    

    # Remove duplicate entries
    df = df.drop_duplicates(['studyID', 't_lab', 'value_name', "value"])
    
    # Convert dates and time into datetime objects
    df.loc[:, 't_lab'] = pd.to_datetime(df['t_lab'])
    df_meta.LOS_hours = pd.to_timedelta(df_meta.LOS_hours)
    
    # Convert data from long into wide format, turning each variable into its
    # own column
    df = df.groupby(['studyID', 't_lab', 'value_name'])['value'].aggregate(
        'first').unstack()
    # Turn time index into a column
    df = df.reset_index(1)
    
    
    # Remove samples that stayed shorter than 24 hours.
    # Because this filtering step is performed using df_meta, which only 
    # contains patients who were acutely admitted to the IC, and thus has a
    # much smaller sample size, this step also effectively filters out patients
    # who were not acutely admitted, which causes a significant drop in sample
    # size.
    df = df[df.index.isin(df_meta[df_meta.LOS_hours > pd.to_timedelta(
        "24:00:00")].index)]
    
    # Add admission date, LOS and readmission to each row in df
    df = df.reset_index().merge(descriptives[
        ["ptId", "icu_admission", "ptLoS"]], right_on = "ptId",
        left_on = "studyID", how = "left").set_index("studyID")
    df.icu_admission = pd.to_datetime(df.icu_admission)
    df.ptLoS = pd.to_timedelta(df.ptLoS)
    # Remove measurements that were taken more than 6 hours before admission
    df = df[df.t_lab >= pd.to_datetime((df.icu_admission - pd.to_timedelta(
        "6:00:00")), utc = True)]
    
    # Exclude medium care patients, and patients before 2013
    if not medium_care:
        exclude_IDs = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS MUMC dataset/toExclude.csv")
        df = df[~df["ptId"].isin(exclude_IDs.toExclude)]
    
    # If time_filter is supplied, remove measurements more than the supplied
    # number of hours since the first measurement.
    if time_filter:
        # Determine first measurement per sample
        first_measurement = pd.DataFrame()
        first_measurement["first_date"] = df.groupby("studyID")[
            "t_lab"].aggregate("min")
        df = pd.merge(df, first_measurement, how = "left", on = "studyID")
        df = df[df.t_lab <= pd.to_datetime((df.first_date + pd.to_timedelta(
            time_filter, unit = "hours")), utc = True)]
    
    # Remove samples admitted after march 14 2020 (to remove COVID patients)
    df = df[df.icu_admission <= pd.to_datetime("2020-03-14")]
    
    # Compute the dismission date of each sample
    df["dismiss_date"] = pd.to_datetime(df.icu_admission + df.ptLoS,
                                        utc = True)
    # Remove measurements that were taken after the computed dismission date.
    df = df[df.t_lab <= df.dismiss_date]

    # Remove unwanted variables
    df.drop(["icu_admission", "t_lab", "ptLoS", "dismiss_date", "ptId"],
            axis = 1, inplace=True)
    if time_filter:
        df.drop("first_date", axis = 1, inplace = True)
    
    # Get LOS in right format
    descriptives.rename(columns = {"ptLoS": "icu_los"}, inplace = True)
    descriptives.icu_los = pd.to_timedelta(descriptives.icu_los)
    descriptives.icu_los = descriptives.icu_los.dt.total_seconds()/60/60/24
    
    # get mean and population standard deviation for each lab variable and
    # store them in seperate columns
    df = df.groupby("studyID").agg(["mean", lambda x: np.std(x)])
    # Set appropriate column names
    colnames = ["" for x in range(df.shape[1])]
    for i, name in enumerate(df.columns):
        if name[1] == '<lambda_0>':
            colnames[i] = name[0] + "_get_var"
        if name[1] == 'mean':
            colnames[i] = name[0] + "_get_mean"
    df.columns = colnames
    df.reset_index(inplace = True)
    
    # Define variables from df_meta that should be included in df
    patient_characteristics_cols = ['age', 'gender', 'apacheIV_score',
                                    'sapsII_score', 'apacheIV_postop1',
                                    'bmi', 'sbp', 'dbp', 'map', 'hr_admission',
                                    'urine_ml_kg_h_6hour', 'cvp',
                                    'mech_vent_vt',
                                    'mech_vent_peep', 'mech_vent_24h',
                                    'mech_vent_admission', 
                                    'mech_vent_resp', 'resp_rate',
                                    'FiO2_low', 'emv_score',
                                    "atrial_fibrillation", "vg_dm", "vg_mi",
                                    "vg_copd", "vg_chroncvdinsuff",
                                    "vg_respinsuff", "vg_dialysis",
                                    "vg_chronnierinsuff", "vg_cirrhose",
                                    "vg_malign_meta", "vg_malign_hema",
                                    "vg_immun_insuff"]
    
    # Add requested patient characteristics to final_df
    df = df.merge(df_meta[patient_characteristics_cols], how = "left",
             on = "studyID")
    
    # Add readmission variable, convert to category  and rename to match SICS
    df = df.merge(descriptives[
        ["ptId", "icu_admission_nr"]], right_on = "ptId",
        left_on = "studyID", how = "left")
    df.rename({"icu_admission_nr" : "prev_admission_icu"}, axis = 1,
              inplace = True)
    df.prev_admission_icu.replace("no readmission", 0, inplace = True)
    df.prev_admission_icu[df.prev_admission_icu != 0] = 1
    df.prev_admission_icu = df.prev_admission_icu.astype("category")
    
    # Convert the central venous blood pressure (cvp) variable into a 
    # categorical variable describing whether cvp was measured or not.
    df["cvp"][df["cvp"].notna()] = 1
    df["cvp"][df["cvp"].isna()] = 0
    
    # Make sure all dataframes contain identical sample rows
    descriptives = descriptives.loc[descriptives.ptId.isin(
        df.studyID),:].reset_index()
    
    # Filter acutely admitted 'critically ill' patients
    # We only keep patients who were ventilated within first 24h or were 
    # connected NOR
    if sample_selection:
        df = df[(df.mech_vent_24h == 1) | (descriptives.nor == 1)]
    
    # Fill some missing valuse
    df["mech_vent_peep"][(df["mech_vent_peep"].isna()) &
                         (df["mech_vent_24h"] == 0)] = 5
    
    # Remove impossible EMV scores
    df.emv_score[(df.emv_score < 3) | (df.emv_score > 15)] = np.nan

    # Clinical examination - respiratory
    df['worsened_resp_cond'] = np.where((
        df['mech_vent_admission'] < df['mech_vent_24h']), 1.0, 0.0)

    # delete variables with > 40% missing
    # df_group = df.groupby(level=0).count().replace({0: np.nan})
    # df = df.loc[:, (df_group.isna().sum() / len(df_group) * 100 > 25.0)]
    # df.loc[:, (df_group.isna().sum() / len(df_group) * 100 > 25.0)].columns
    initial_vars = df.columns
    df.dropna(axis = 1, thresh = 0.6*len(df), inplace = True)
    print("Variables removed due to excessive missingness:")
    print(initial_vars[~initial_vars.isin(df.columns)].values)
    
    # For variables with <=5% missingness, drop the missing samples
    # varmis_5 = df.loc[:,df.isna().sum()/len(df) <=0.01].columns
    # for var in varmis_5:
    #     df = df.loc[~df.loc[:, var].isna(),:]
    
    
    # delete IDs with > 40% missing
    initial_samples = len(df)
    df.dropna(axis = 0, thresh = 0.6*df.shape[1], inplace = True)
    print(f"{initial_samples - len(df)} samples were removed due to excessive"\
          " missingness.")
    
    # Adjust unit of urine output during first 6 hours
    df.urine_ml_kg_h_6hour = df.urine_ml_kg_h_6hour/1000
    
    # Only keep samples in descriptives for which lab data is available
    descriptives = descriptives[descriptives.ptId.isin(df.studyID)]
    
    # Store mortality per sample
    # Compute number of days between discharge and death
    descriptives["dismiss_date"] = pd.to_datetime(
        pd.to_datetime(descriptives.icu_admission) +
        pd.to_timedelta(descriptives.icu_los, unit = "days"),
        utc = True)
    descriptives.date_of_decease = pd.to_datetime(descriptives.date_of_decease,
                                                  utc = True,
                                                  format = "%d-%m-%Y %H:%M:%S")
    descriptives["mort_days"] = pd.to_timedelta(descriptives.date_of_decease - 
                                                descriptives.dismiss_date,
                                                unit = "d").dt.round(
                                                    freq='d').dt.days
    descriptives["mort_days"][descriptives["mort_icu"] == 1] = 0
    
    print("Mortality per 30/90 days not yet included!")
    y_mort = descriptives.set_index("ptId")["mort_icu"]
    descriptives["mort30"] = descriptives.mort_days <=30
    descriptives["mort30"][descriptives.mort_days.isna()] = np.nan
    descriptives["mort90"] = descriptives.mort_days <=90
    descriptives["mort90"][descriptives.mort_days.isna()] = np.nan
    
    # Store aki per sample
    print("aki not yet included!")
    y_aki = None
    
    # Store Variable names
    var_arr = list(df.columns)
    
    # Remove outliers
    df['hr_admission'][df['hr_admission'] == 0] = np.nan
    
    # Remove ptId variable
    df.drop("ptId", axis = 1, inplace = True)
    
    
    # If scaling was requested, return scaled data, otherwise unscaled data
    if(scale_data):
        # Scale the variables with a min max scaler
        # df = pd.DataFrame(MinMaxScaler().fit_transform(
        #     df.drop("studyID", axis = 1)), columns = df.columns[1:])
        
        # Temporarily drop categorical features
        cat_features = ["studyID",'gender', 'apacheIV_postop1', 'cvp',
                     'mech_vent_24h', 'mech_vent_admission',
                     'worsened_resp_cond']
        df_numeric = df.drop(cat_features, axis = 1)
        
        df_numeric = pd.DataFrame(StandardScaler().fit_transform(
            df_numeric), columns = df_numeric.columns)
        
        # Combine scaled numeric features with unscaled categorical features
        df = pd.concat([df.reset_index()[cat_features], df_numeric], axis = 1)
        
        # Set studyID as index
        df.set_index("studyID", inplace = True)
        print("returning scaled data")
    else:
        df.set_index("studyID", inplace = True)
        print("returning unscaled data")
    
    # Set columns to appropriate dtype
    df = set_dtypes(df)
    
    # Set descriptives index
    descriptives.set_index("index", inplace = True)
    
    return df, y_mort, y_aki, var_arr, descriptives
    


def load_sics_data(seed = 5192, pca = False, scale_data = False,
                   var_filter = None, binary_cvp = True, time_filter = None):
    """
    load_sics_data loads and prepares the data from sics for clustering analysis.

    Parameters
    ----------
    seed : integer, optional
        integer specifying the random seed for reproducible random operations.
        The default is 5192.
    pca : boolean, optional
        Boolean indicating whether or not to perform pca. The default is False.
    scale_data : boolean, optional
        Boolean indicating whether to scale the data with a robust scaler or
        not. The default is False.
    

    Returns
    -------
    x : numpy.ndarray
        The data matrix containing all samples and variables for the custering
        analysis.
    y_mort : numpy.ndarray
        Dependent variables array regarding mortality of the samples.
    y_aki : numpy.ndarray
        Dependent variables array regarding aki of the samples.
    var_arr : list
        List containing the names of all variables in x.
    descriptives : DataFrame
        Dataframe containing additional information of the samples.
    """
    
    # Load the main data file
    df = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/final_merged.csv", encoding= 'unicode_escape')
    # Only keep specific variables
    df = df[~df.value_name.isin(['APTT (oud) (BL)', 'Glucose (BL)',
                                 'POC art Base Excess', 'PT (oud) (BL)'])]
    # Remove duplicate entries
    df = df.drop_duplicates(['studyID', 't_lab', 'value_name', "value_id",
                             "value", "unit", "date_match"])
    # Convert dates into datetime objects
    df.loc[:, 't_lab'] = pd.to_datetime(df['t_lab'])
    df.loc[:, 'date_match'] = pd.to_datetime(df['date_match'])
    
    # Load additional information about study subjects
    base = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/extradata_SICS.csv", index_col=0)
    
    # Deduce indices of sample with missing values on aki
    nan_ids = base["studyID"][(base["SICS.aki1"] == -9999) |
                             ((base["SICS.aki2"] == -9999) &
                              (base["SICS.aki1"] == 1)) |
                             ((base["SICS.aki3"] == -9999) &
                              (base["SICS.aki2"] == 1))]
    nan_ids = list(nan_ids)
    
    
    # Replace -9999 values with np.nan
    base = base.replace(-9999, np.nan)
    
    # Store the sum of aki 1,2 and 3 in a seperate variable
    base["aki"] = base[["SICS.aki1", "SICS.aki2", "SICS.aki3"]].sum(axis=1)
    base["aki"].loc[base["studyID"].isin(nan_ids)] = np.nan

    # Compute delta T
    base["deltaT"] = base["temp_centr"] - ((base["temp_periph_1"] +
                                            base["temp_periph_2"]) / 2)
    
    # convert mottling into a categorical dummy variable
    mottling = pd.get_dummies(base["mottling"], prefix="mottling")
    base = pd.concat([base, mottling], axis=1)
    
    # Filter out specific variables
    base = base[
        ["studyID", "LOS_hours", "age", "gender", "height", "weight", "resp_rate", "mech_vent", "atrial_fibrillation",
         "heart_rhythm", "mottling_0", "mottling_1", "mottling_2", "mottling_3", "mottling_4", "mottling_5", "deltaT",
         "avpu", "vasoactive", "ci_cl"]]
    
    # delete IDs with base < 24
    IDs = base["studyID"][base["LOS_hours"] >= 24]
    sics = df[df["studyID"].isin(IDs)]

    # transform table
    sics = sics.groupby(['studyID', 't_lab', 'value_name'])['value'].aggregate(
        'first').unstack()
    sics = sics.reset_index(1)

    # compute relative time (hours) since admission and delete measurements after base
    sics["Reltime"] = sics.groupby(["studyID"])["t_lab"].apply(
        lambda x:convert_to_reltime(x, base))
    # remove t_lab column
    sics = sics.drop(columns=["t_lab"])
    # Drop measurements which were taken after the LOS
    sics = sics.loc[sics["Reltime"] != -1]
    
    # If time filter, only take time_filter hours since first lab measurement
    if time_filter != None:
        sics = sics.loc[sics["Reltime"] <= time_filter]

    # delete variables with > 40% missing
    sics_group = sics.groupby(level=0).count().replace({0: np.nan})
    sics = sics.loc[:, (sics_group.isna().sum() / len(sics_group) * 100 <= 40.0)]

    # delete IDs with > 40% missing
    sics_del = sics.loc[sics_group.isna().sum(axis=1) / (len(list(sics_group)) - 2) * 100 < 40, :]
    sics = sics.reset_index()
    sics_del = list(sics_del.index.unique())
    sics = sics.loc[sics['studyID'].isin(sics_del), :]

    print(len(list(sics['studyID'].unique())))

    # Drop the Reltime column
    sics = sics.drop(columns=["Reltime"])
    # Print all remaining variable names
    print(list(sics))
    
    # Load apache diagnosis information
    data = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/SICS1_apache_diagnoses.csv", index_col=0,
                       engine='python')
    
    # Remove the aids variable (vg_aids)
    #data.drop("vg_aids", axis = 1, inplace=True)
    
    # replace missing data with NaN
    data = data.replace('none', np.nan)
    data = data.replace('.a', np.nan)
    data = data.replace('None', np.nan)

    # determine mortality
    data['mort_icu'] = np.where(((data['mort_icu'].isnull()) &
                                 (data['mort_date'].isnull())), 'no',
                                data['mort_icu'])
    data['mort90'] = np.where(((data['mort90'].isnull()) &
                               (data['mort_date'].isnull())), 'no',
                              data['mort90'])
    data['mort30'] = np.nan
    data['mort30'] = data.apply(calculate_mort30, axis=1)

    # heart rhythm
    data['heart_rhythm'] = data['heart_rhythm'].replace({'regular': 1,
                                                         'irregular': 0})

    # gender
    genders = {'male': 1, 'female': 0}
    data['gender'] = np.where(data.gender.isin(genders.keys()),
                              data.gender.map(genders), np.nan).astype(int)

    # age
    data['age'] = pd.to_numeric(data['age'], errors='coerce', downcast='integer').round()
    
    binary_descriptives = ['mech_vent', 'mech_vent_admission', 'mech_vent_24h',
                           'vg_mi', 'vg_dm', 'vg_chroncvdinsuff',
                           'vg_copd', 'vg_respinsuff', 'vg_chronnierinsuff',
                           'vg_dialysis', 'vg_cirrhose', 'vg_malign_meta',
                           'vg_malign_hema','vg_aids', 'vg_immun_insuff',
                           'atrial_fibrillation', 'apacheIV_postop1',
                           'cardiogenic', 'distributive', 'hypovolemic',
                           'obstructive']
    apaches = ['apacheII_postop', 'apacheII_subgroup', 'apacheIV_postop1',
               'apacheIV_postop2', 'apacheIV_subgroup1',
               'apacheIV_subgroup2']
    for apache in apaches:
        data[apache] = data[apache].str.extract('(\D+)', expand=False)
    for descriptive in binary_descriptives:
        data[descriptive] = data[descriptive].replace({'yes': 1, 'no': 0})
    

    # Initiate final_df
    final_df = pd.DataFrame()

    # Set requested patient characteristics
    patient_characteristics_cols = ['age', 'gender', 'apacheIV_score',
                                    'sapsII_score', 'apacheIV_postop1',
                                    'cardiogenic', 'distributive',
                                    'hypovolemic', 'obstructive']
    
    # Add requested patient characteristics to final_df
    final_df[patient_characteristics_cols] = data[patient_characteristics_cols]

    # Compute bmi
    final_df['bmi'] = (data['weight'] / ((data['height'] / 100) ** 2)).round(2)
    # Add variable regarding whether a patient has been readmitted
    final_df['prev_admission_icu'] = np.where(data['icu_admission_nr'] ==
                                              'no readmission', 0, 1)

    # Clinical examination - hemodynamics
    final_df['cardiac_index_normal'] = np.where((
        data['co_cl'] /np.sqrt(data['height'] * data['weight'] / 3600) > 2.2),
        1, 0)
    final_df['mottling'] = np.where((data['mottling'] > 3), 1, 0)
    final_df['crt'] = np.where(((data['crt_sternum'] > 4.5) |
                                (data['crt_knee'] > 4.5) |
                                (data['crt_finger'] > 4.5)), 1, 0)
    final_df['sbp'] = data[['sbp1', 'sbp2']].mean(axis=1)
    final_df['dbp'] = data[['dbp1', 'dbp2']].mean(axis=1)
    final_df['map'] = data[['map1', 'map2']].mean(axis=1)
    # Add hemodynamic variables from data to final_df
    hemodynamics = ['atrial_fibrillation', 'hr_admission',
                    'urine_ml_kg_h_6hour', 'cvp']
    final_df[hemodynamics] = data[hemodynamics]
    
    # Convert the central venous blood pressure (cvp) variable into a 
    # categorical variable describing whether cvp was measured or not.
    if binary_cvp:
        final_df["cvp"][final_df["cvp"].notna()] = 1
        final_df["cvp"][final_df["cvp"].isna()] = 0

    # Clinical examination - respiratory
    final_df['worsened_resp_cond'] = np.where((
        data['mech_vent_admission'] <data['mech_vent_24h']), 1.0, 0.0)
    
    # Add respiratory variables from data to final_df
    respiratory = ['mech_vent_vt', 'mech_vent_resp', 'mech_vent_peep',
                   'mech_vent_24h', 'mech_vent_admission',
                   'resp_rate', 'FiO2_low']
    final_df[respiratory] = data[respiratory]

    # Clinical examination - other
    data['emv_score'] = np.nan
    data['emv_score'] = data.apply(calculate_emv_score, axis=1)
    final_df['emv_score'] = data['emv_score']

    # Add variables regarding history of medical conditions from data to final_df
    final_df = final_df.join(data[data.columns[202:214]])
    # Turn index into column so it can be saved to csv
    final_df['studyID'] = data.index
    
    # Write final df to disk
    #final_df.to_csv('stats/analysis.csv')
    
    # Rename final_df to analysis
    analysis = final_df
    # Remove studyID column
    analysis = analysis.drop(columns=['studyID'])


    # get mean and varirance for each variable in sics and store them in
    # seperate columns
    sics = sics.groupby('studyID').agg([get_mean, get_var])
    sics.columns = ["_".join(x) for x in sics.columns.ravel()]
    sics = sics.reset_index(level=0)
    sics = sics.replace({-1.0: np.nan})
    
    

    # if pca, then add pca scores of first two components to sics, otherwise
    # merge sics with analysis without PCA scores and write to disk.
    if pca is True:
        # impute the data
        scaler = MinMaxScaler()
        data_rescaled = scaler.fit_transform(analysis)
        imp = IterativeImputer(max_iter=10, random_state=seed)
        data_rescaled = imp.fit_transform(data_rescaled)
        
        principal_components = PCA(n_components=2).fit_transform(data_rescaled)
        
        principal_df = pd.DataFrame(data=principal_components,
                                    columns=['principal component 1',
                                             'principal component 2'])
        principal_df['studyID'] = analysis.reset_index(0)['studyID']

        sics = pd.merge(sics, principal_df, on='studyID')
    else:
        sics = pd.merge(sics, analysis, on='studyID')
    #sics.to_csv('stats/sics.csv')

    # Load mortality information per studyID and include in sics
    mort = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/extradata_SICS.csv")
    mort = mort[["studyID", "mortality"]]
    sics = sics.merge(mort, on="studyID")

    # Load additional information 
    descriptives = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/descriptives_SICS.csv", index_col=0)
    descriptives["aki"] = descriptives[["aki1", "aki2", "aki3"]].sum(axis=1)
    descriptives["aki0"] = (descriptives["aki"] == 0).astype(int)
    descriptives["aki1"] = (descriptives["aki"] == 1).astype(int)
    descriptives["aki2"] = (descriptives["aki"] == 2).astype(int)
    descriptives["aki3"] = (descriptives["aki"] == 3).astype(int)


    # Load additional information regarding aki
    df_aki = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/extradata_SICS.csv")
    df_aki = df_aki[["studyID", "SICS.aki1", "SICS.aki2", "SICS.aki3"]]

    deleted_ids = df_aki["studyID"][
        (df_aki["SICS.aki1"] == -9999) | ((df_aki["SICS.aki2"] == -9999) &
                                          (df_aki["SICS.aki1"] == 1)) | (
                (df_aki["SICS.aki3"] == -9999) & (df_aki["SICS.aki2"] == 1))]
    deleted_ids = list(deleted_ids)
    # df_aki = df_aki[df_aki["studyID"].isin(deleted_ids)==False]

    df_aki = df_aki.replace(-9999, 0)
    df_aki.loc[:, "AKI_val"] = df_aki[["SICS.aki1", "SICS.aki2",
                                       "SICS.aki3"]].sum(axis=1)
    df_aki = df_aki.drop(columns=["SICS.aki1", "SICS.aki2", "SICS.aki3"])
    df_aki.loc[df_aki["studyID"].isin(deleted_ids), 'AKI_val'] = np.nan

    # Merge additional aki information with sics
    sics = sics.merge(df_aki, on="studyID")
    
    # Store descriptives information for each row in sics
    descriptives = descriptives.merge(sics["studyID"], on="studyID")

    # Isolate the two outcome variables
    y_mort = sics["mortality"].values
    y_aki = sics["AKI_val"].values
    
    # Remove unnesecary columns
    x = sics.drop(columns=["mortality", "AKI_val"])
    descriptives = descriptives.drop(columns=["studyID"])
    
    # Store variable names
    var_arr = list(x)
    
    # If scaling was requested, return scaled data, otherwise unscaled data
    if(scale_data):
        # Scale the variables with a robust scaler
        #df = pd.DataFrame(RobustScaler().fit_transform(x), columns = x.columns)
        #df = pd.DataFrame(MinMaxScaler().fit_transform(x), columns = x.columns)
        
        # Temporarily drop categorical features
        cat_features = ["studyID",'gender', 'apacheIV_postop1', 'cvp',
                     'mech_vent_24h', 'mech_vent_admission',
                     'worsened_resp_cond']
        x_numeric = x.drop(cat_features, axis = 1)
        
        x_numeric = pd.DataFrame(StandardScaler().fit_transform(
            x_numeric), columns = x_numeric.columns)
        
        # Combine scaled numeric features with unscaled categorical features
        df = pd.concat([x[cat_features], x_numeric], axis = 1)
        print("returning scaled data")
    else:
        df = x.copy()
        print("returning unscaled data")
    
    # Add information about the discharge diagnosis
    discharge_diagnoses = pd.read_csv("L:/SPEC/ICU/RESEARCH/Data-onderzoek/SICS Groningen dataset/discharge_diagnoses.csv")
    descriptives["discharge diagnosis"] = discharge_diagnoses[
        "Discharge diagnosis"]
    
    # Set studyID as index
    df.set_index("studyID", inplace = True)
    descriptives.index = df.index
    
    # Apply variable filter if requested
    if not var_filter == None:
        df = df[var_filter]
    
    # Manually fix two missing apacheIV_postop1 values
    df.loc[289, "apacheIV_postop1"] = 1
    descriptives.loc[289, "apacheIV_postop1"] = 1
    
    # # Set columns to appropriate dtype
    df = set_dtypes(df)
    
    return df, y_mort, y_aki, var_arr, descriptives
    
    
    # normalize a series using a given scaler
def normalize_column(series, scaler):
    series = series.values.reshape(-1, 1)
    s_scaled = scaler.fit_transform(series)
    s_scaled = s_scaled.reshape((-1))
    return s_scaled
    
def convert_to_reltime(t_lab, los):
    start = t_lab.min()
    t_lab = t_lab - start
    t_lab = t_lab / np.timedelta64(1, 'h')
    max_hours = los[los['studyID'] == t_lab.name]['LOS_hours'].iloc[0]
    t_lab = t_lab.apply(lambda x: x if x <= max_hours else -1)

    return t_lab

def calculate_mort30(row):
    date_admission_format = '%m/%d/%Y %H:%M'
    date_death_format = '%m/%d/%Y'
    admission_date = convert_date(row['admission_dt_hospital'],
                                  date_admission_format)
    death_date = convert_date(row['mort_date'], date_death_format)
    death_90 = row['mort90']
    if death_90 == 'no':
        row['mort30'] = 'no'
    elif death_90 == 'yes':
        if death_date is np.nan or admission_date is np.nan:
            row['mort30'] = np.nan
        elif (death_date - admission_date).days <= 30:
            row['mort30'] = 'yes'
        else:
            row['mort30'] = 'no'
    return row['mort30']
    
def convert_date(date, dateformat):
    try:
        if date is not np.nan and pd.notna(date):
            converted = datetime.datetime.strptime(date, dateformat)
        else:
            converted = np.nan
    except ValueError:
        converted = np.nan
    return converted

def calculate_emv_score(row):
    total_score = 0
    eyes_score = {"spontaneously": 4, "verbal": 3, "pain": 2}
    verbal_score = {"oriented": 5, "confused": 4, "inappropriate": 3,
                    "incomprehensible": 2}
    motor_score = {"obeys": 6, "localizes": 5, "withdraws": 4, "flexion": 3,
                   "extension": 2}
    if row['emv_e'] is not None and row['emv_e'] is not np.nan:
        total_score += eyes_score[row['emv_e']]
    if row['emv_v'] is not None and row['emv_v'] is not np.nan:
        total_score += verbal_score[row['emv_v']]
    if row['emv_m'] is not None and row['emv_m'] is not np.nan:
        total_score += motor_score[row['emv_m']]
    if total_score == 0:
        total_score = 3
    row['emv_score'] = total_score
    return row['emv_score']

def get_mean(group):
    if np.issubdtype(group.dtype, np.number):
        return np.nanmean(group)

    last_index = group.last_valid_index()
    return group[last_index] if last_index else -1


def get_var(group):
    if np.issubdtype(group.dtype, np.number):
        return np.nanstd(group)

    last_index = group.last_valid_index()
    return group[last_index] if last_index else -1


def compute_missingness(df, showit = False, threshold = 5,  save = False,
                        savename = "degrees of missingness",
                        filetype = "svg"):
    """
    compute_missingness computes the degree of missingness per variable and per
    sample.

    Parameters
    ----------
    df : matrix or dataframe
        rows represent samples and columns represent variables.
    showit : boolean, optional
        indicates whether to  plot of missingness results. Default is False.
    threshold : integer, optional.
        specifies the minimum percentage of missingness for a variable to
        be included in the plot of variable missingness if showit = True.
        Default is 5.
    save : boolean, optional
        boolean indicating whether to locally save the plot. The default is
        True.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        DESCRIPTION. The default is "svg".

    Returns
    -------
    variable_missingness : DataFrame
        rows represent variables and columns represent variable name, for how
        many samples this variable was missing, and the percentage of samples
        for which the variable was missing respectively.
    sample_missingness : DataFrame
        rows represent samples and columns represent sample name, for how
        many variables this sample had missing values, and the percentage of
        variables for which the sample was missing respectively.

    """
    # Check if there are any missing values
    if not df.isna().any().any():
        raise Exception("There are no missing values in the data.")
    
    # Initiate dataframe to store variable missingness results
    variable_missingness = pd.DataFrame(np.full((df.shape[1],3), np.nan))
    variable_missingness.columns = ["variable", "count", "percentage"]
    # Fill the columns with the desired information
    for i, col in enumerate(df.columns):
        variable_missingness.iloc[i, 0] = col
        variable_missingness.iloc[i, 1] = df[col].isna().sum()
        variable_missingness.iloc[i, 2] = df[col].isna().sum()/len(df)*100
    
    # Initiate dataframe to store sample missingess information
    sample_missingness = pd.DataFrame(np.full((df.shape[0], 2),
                                              np.nan)).reset_index()
    sample_missingness.columns = ["sample", "count", "percentage"]
    sample_missingness['sample'] = sample_missingness['sample'].astype(str)
    # Fill the columns with the desired information
    for i in range(len(df)):
        sample_missingness.iloc[i, 1] =  df.iloc[i,:].isna().sum()
        sample_missingness.iloc[i, 2] =  df.iloc[i,:].isna().sum(
            )/df.shape[1]*100
        
    # If requested, plot the results.
    if showit:
        variable_plot = variable_missingness[
            variable_missingness.percentage >= threshold]
        plt.figure()
        ax = sns.barplot(x = "percentage", y = "variable",
                     data = variable_plot.sort_values(
                         "percentage", ascending = False))
        ax.set_title("Missingness percentage per variable")
        ax.set_xlabel("Percentage of missing values")
        ax.set_ylabel("")
        plt.tight_layout()
        
        if(save):
            plt.savefig(f'figures/{savename}_variables.{filetype}',
                    bbox_inches='tight')
            
        plt.figure()
        ax = sns.histplot(x = "percentage", data = sample_missingness,
                          kde = True, stat="percent", binwidth = 1)
        ax.set_title("Missingness percentage per sample")
        ax.set_xlabel("Percentage of missing values")
        ax.set_ylabel("Percentage of samples")
        plt.tight_layout()
        
        if(save):
            plt.savefig(f'figures/{savename}_samples.{filetype}',
                    bbox_inches='tight')
        
        
    return variable_missingness, sample_missingness
    
def set_dtypes(df):
    '''
    Set appropriate dtypes to input dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of input data.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of input data with appropriate data types.

    '''
    # Specify categroical variables
    categorical_variables = ['gender', 'apacheIV_postop1',
                             'atrial_fibrillation', 'cvp', 'mech_vent_24h',
                             'mech_vent_admission',
                             'worsened_resp_cond', 'prev_admission_icu',
                             "vg_copd", "vg_chroncvdinsuff", "vg_respinsuff",
                             "vg_dialysis", "vg_chronnierinsuff",
                             "vg_cirrhose", "vg_malign_meta", "vg_malign_hema",
                             "vg_immun_insuff", "vg_mi", "vg_dm"]
    # specify integer variables
    integer_variables = ['age', 'apacheIV_score', 'sapsII_score',
                         'hr_admission', 'resp_rate',
                         'FiO2_low', 'emv_score']
    
    # Cast categorical variables to the category dtype
    df[categorical_variables] = df[categorical_variables].astype('category')
    
    return(df)
    

def scale_data(df, method = "minmax"):
    '''
    Scales a dataset in the specified manner.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data that should be scaled. Make sure all categorical
        variables are binary, and of the 'category' dtype.
    method : str, optional
        Specifies which form of scaling to apply. The default is "minmax".
            minmax : performs min max scaling, forcing all variables to a range
                between 0 and 1.
            standardise : performs standardisation to a Z-distribution
            log : performs log scaling.
            robustscaler : applies the RobustScaler from the scikitlearn pacakge

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the scaled data.
    scaler : function
        A function that can be used to apply the (trained) scaler to new 
        datasets.

    '''
    if str.lower(method)=="minmax":
        scaler = MinMaxScaler()
        df = pd.DataFrame(scaler.fit_transform(df),
                          columns = df.columns)
    elif str.lower(method) == "standardise" or method == "standardize":
        scaler = StandardScaler()
        cat_features = df.columns[df.dtypes == "category"]
        df_numeric = df.drop(cat_features, axis = 1)
        df_numeric = pd.DataFrame(scaler.fit_transform(
            df_numeric), columns = df_numeric.columns)
        # Combine scaled numeric features with unscaled categorical features
        df = pd.concat([df.reset_index()[cat_features], df_numeric],
                       axis = 1).set_index(df.index)
    elif str.lower(method) == "log":
        # Log transform all labe measurements (any variable either '_mean' or
        # '_var' in the name)
        cat_features = df.loc[:,~((df.columns.str.contains("_var")) |
                              (df.columns.str.contains("_mean")))].columns
        df_numeric = df.drop(cat_features, axis = 1)
        df_numeric = np.log(df_numeric +1)
        # Combine scaled numeric features with unscaled categorical features
        df = pd.concat([df.reset_index()[cat_features],
                        df_numeric.reset_index(drop=True)],
                       axis = 1).set_index(df.index)
        scaler = np.log
    elif str.lower(method) == "robustscaler":
        scaler = RobustScaler()
        cat_features = df.columns[df.dtypes == "category"]
        df_numeric = df.drop(cat_features, axis = 1)
        df_numeric = pd.DataFrame(scaler.fit_transform(
            df_numeric), columns = df_numeric.columns)
        # Combine scaled numeric features with unscaled categorical features
        df = pd.concat([df.reset_index()[cat_features], df_numeric],
                       axis = 1).set_index(df.index)
    else:
       raise(ValueError(f"'{method}' method not recognised"))
        
        
    return df, scaler
    