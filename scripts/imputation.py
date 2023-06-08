# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:24:12 2022

@author: Jip de Kok
"""

# Load packages
import miceforest as mf
import matplotlib.pyplot as plt
import pandas as pd

# Load functions from local scripts
from miceforest import mean_matching_functions as mmf
from data_functions import compute_missingness
import time
import numpy as np

# Load function for original imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_data(df, df_mumc, showit = False, seed = 5192,
                n_imputations = 100, n_opt_steps = 5, n_iterations = 5):
    '''
    Performs miceforest imputation the SICS and MUMC+ datasets. Van also
    generate some insightful figures.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the SICS input data set.
    df_mumc : pandas.DataFrame
        DataFrame containing the MUMC+ input data set.
    showit : boolean, optional
        Boolean indicating whether to produce figures. The default is False.
    seed : integer, optional
        Random seed. The default is 5192.
    n_imputations : int, optional
        Number of imputations to perform. The default is 100.
    n_opt_steps : int, optional
        Number of optimisation steps to perform. The default is 5.
    n_iterations : int, optional
        The number of iterations to run. The default is 5.

    Returns
    -------
    df : pandas.DataFrame
        Imputed SICS dataset.
    df_mumc : pandas.DataFrame
        Imputed MUMC+ dataset.
    kernel_sics : miceforest.ImputationKernel
        Kernel for imputing the SICS dataset.
    kernel_mumc : miceforest.ImputationKernel
        Kernel for imputing the MUMC+ dataset.

    '''
    
    if showit:
        var_mis, sam_mis = compute_missingness(df, showit = False,
                                               threshold = 0.01)
        var_mis_mumc, sam_mis_mumc = compute_missingness(df_mumc,
                                                         showit = False)
        
    # Define variablse on which to apply predictive mean matching (PMM)
    var_mmc = {
        'gender'                : 10,
        'apacheIV_postop1'      : 10,
        'atrial_fibrillation'   : 10,
        'cvp'                   : 10,
        'mech_vent_24h'         : 10,
        'mech_vent_admission'   : 10,
        'mech_vent_resp'        : 10,
        'worsened_resp_cond'    : 10,
        'prev_admission_icu'    : 10,
        'age'                   : 10,
        'apacheIV_score'        : 10,
        'sapsII_score'          : 10,
        'hr_admission'          : 10,
        'resp_rate'             : 10,
        'FiO2_low'              : 10,
        'emv_score'             : 10,
        'vg_mi'                 : 10,
        'vg_dm'                 : 10,
        'vg_chroncvdinsuff'     : 10,
        'vg_copd'               : 10,
        'vg_respinsuff'         : 10,
        'vg_chronnierinsuff'    : 10,
        'vg_dialysis'           : 10,
        'vg_cirrhose'           : 10,
        'vg_malign_meta'        : 10,
        'vg_malign_hema'        : 10,
        'vg_immun_insuff'       : 10}
    
    var_sch = {
         'ALAT (BL)_get_mean': df.columns[~df.columns.isin(
             ["ALAT (BL)_get_var", "ALAT (BL)_get_mean"])].values,
         'ALAT (BL)_get_var': df.columns[~df.columns.isin(
             ["ALAT (BL)_get_mean", "ALAT (BL)_get_var"])].values,
         'ASAT (BL)_get_mean': df.columns[~df.columns.isin(
             ["ASAT (BL)_get_var", "ASAT (BL)_get_mean"])].values,
         'ASAT (BL)_get_var': df.columns[~df.columns.isin(
             ["ASAT (BL)_get_var", "ASAT (BL)_get_mean"])].values,
         'Albumine (BL)_get_mean': df.columns[~df.columns.isin(
             ["Albumine (BL)_get_var", "Albumine (BL)_get_mean"])].values,
         'Albumine (BL)_get_var': df.columns[~df.columns.isin(
             ["Albumine (BL)_get_var", "Albumine (BL)_get_mean"])].values,
         'Alkalische fosfatase (BL)_get_mean': df.columns[~df.columns.isin(
             ["Alkalische fosfatase (BL)_get_var", "Alkalische fosfatase (BL)_get_mean"])].values,
         'Alkalische fosfatase (BL)_get_var': df.columns[~df.columns.isin(
             ["Alkalische fosfatase (BL)_get_var", "Alkalische fosfatase (BL)_get_mean"])].values,
         'Bilirubine totaal (BL)_get_mean': df.columns[~df.columns.isin(
             ["Bilirubine totaal (BL)_get_var", "Bilirubine totaal (BL)_get_mean"])].values,
         'Bilirubine totaal (BL)_get_var': df.columns[~df.columns.isin(
             ["Bilirubine totaal (BL)_get_var", "Bilirubine totaal (BL)_get_mean"])].values,
         'CK (BL)_get_mean': df.columns[~df.columns.isin(
             ["CK (BL)_get_var", "CK (BL)_get_mean"])].values,
         'CK (BL)_get_var': df.columns[~df.columns.isin(
             ["CK (BL)_get_var", "CK (BL)_get_mean"])].values,
         'CRP (BL)_get_mean': df.columns[~df.columns.isin(
             ["CRP (BL)_get_var", "CRP (BL)_get_mean"])].values,
         'CRP (BL)_get_var': df.columns[~df.columns.isin(
             ["CRP (BL)_get_var", "CRP (BL)_get_mean"])].values,
         'Calcium (BL)_get_mean': df.columns[~df.columns.isin(
             ["Calcium (BL)_get_var", "Calcium (BL)_get_mean"])].values,
         'Calcium (BL)_get_var': df.columns[~df.columns.isin(
             ["Calcium (BL)_get_var", "Calcium (BL)_get_mean"])].values,
         'Chloride (BL)_get_mean': df.columns[~df.columns.isin(
             ["Chloride (BL)_get_var", "Chloride (BL)_get_mean"])].values,
         'Chloride (BL)_get_var': df.columns[~df.columns.isin(
             ["Chloride (BL)_get_var", "Chloride (BL)_get_mean"])].values,
         'Fibrinogeen (BL)_get_mean': df.columns[~df.columns.isin(
             ["Fibrinogeen (BL)_get_var", "Fibrinogeen (BL)_get_mean"])].values,
         'Fibrinogeen (BL)_get_var': df.columns[~df.columns.isin(
             ["Fibrinogeen (BL)_get_var", "Fibrinogeen (BL)_get_mean"])].values,
         'Fosfaat (BL)_get_mean': df.columns[~df.columns.isin(
             ["Fosfaat (BL)_get_var", "Fosfaat (BL)_get_mean"])].values,
         'Fosfaat (BL)_get_var': df.columns[~df.columns.isin(
             ["Fosfaat (BL)_get_var", "Fosfaat (BL)_get_mean"])].values,
         'Gamma-GT (BL)_get_mean': df.columns[~df.columns.isin(
             ["Gamma-GT (BL)_get_var", "Gamma-GT (BL)_get_mean"])].values,
         'Gamma-GT (BL)_get_var': df.columns[~df.columns.isin(
             ["Gamma-GT (BL)_get_var", "Gamma-GT (BL)_get_mean"])].values,
         'Hb (BL)_get_mean': df.columns[~df.columns.isin(
             ["Hb (BL)_get_var", "Hb (BL)_get_mean", "Ht (BL)_get_var",
              "Ht (BL)_get_mean"])].values,
         'Hb (BL)_get_var': df.columns[~df.columns.isin(
             ["Hb (BL)_get_var", "Hb (BL)_get_mean", "Ht (BL)_get_var",
              "Ht (BL)_get_mean"])].values,
         'Ht (BL)_get_mean': df.columns[~df.columns.isin(
             ["Ht (BL)_get_var", "Ht (BL)_get_mean", "Hb (BL)_get_var",
              "Hb (BL)_get_mean"])].values,
         'Ht (BL)_get_var': df.columns[~df.columns.isin(
             ["Ht (BL)_get_var", "Ht (BL)_get_mean", "Hb (BL)_get_var",
              "Hb (BL)_get_mean"])].values,
         'Kalium (BL)_get_mean': df.columns[~df.columns.isin(
             ["Kalium (BL)_get_var", "Kalium (BL)_get_mean"])].values,
         'Kalium (BL)_get_var': df.columns[~df.columns.isin(
             ["Kalium (BL)_get_var", "Kalium (BL)_get_mean"])].values,
         'Kreatinine (BL)_get_mean': df.columns[~df.columns.isin(
             ["Kreatinine (BL)_get_var", "Kreatinine (BL)_get_mean"])].values,
         'Kreatinine (BL)_get_var': df.columns[~df.columns.isin(
             ["Kreatinine (BL)_get_var", "Kreatinine (BL)_get_mean"])].values,
         'LDH (BL)_get_mean': df.columns[~df.columns.isin(
             ["LDH (BL)_get_var", "LDH (BL)_get_mean"])].values,
         'LDH (BL)_get_var': df.columns[~df.columns.isin(
             ["LDH (BL)_get_var", "LDH (BL)_get_mean"])].values,
         'Leukocyten (BL)_get_mean': df.columns[~df.columns.isin(
             ["Leukocyten (BL)_get_var", "Leukocyten (BL)_get_mean"])].values,
         'Leukocyten (BL)_get_var': df.columns[~df.columns.isin(
             ["Leukocyten (BL)_get_var", "Leukocyten (BL)_get_mean"])].values,
         'Magnesium (BL)_get_mean': df.columns[~df.columns.isin(
             ["Magnesium (BL)_get_var", "Magnesium (BL)_get_mean"])].values,
         'Magnesium (BL)_get_var': df.columns[~df.columns.isin(
             ["Magnesium (BL)_get_var", "Magnesium (BL)_get_mean"])].values,
         'Natrium (BL)_get_mean': df.columns[~df.columns.isin(
             ["Natrium (BL)_get_var", "Natrium (BL)_get_mean"])].values,
         'Natrium (BL)_get_var': df.columns[~df.columns.isin(
             ["Natrium (BL)_get_var", "Natrium (BL)_get_mean"])].values,
         'Trombocyten (BL)_get_mean': df.columns[~df.columns.isin(
             ["Trombocyten (BL)_get_var", "Trombocyten (BL)_get_mean"])].values,
         'Trombocyten (BL)_get_var': df.columns[~df.columns.isin(
             ["Trombocyten (BL)_get_var", "Trombocyten (BL)_get_mean"])].values,
         'Ureum (BL)_get_mean': df.columns[~df.columns.isin(
             ["Ureum (BL)_get_var", "Ureum (BL)_get_mean",
              "Albumine (BL)_get_var", "Albumine (BL)_get_mean",
              "Calcium (BL)_get_var", "Calcium (BL)_get_mean",
              "Chloride (BL)_get_var", "Chloride (BL)_get_mean",
              "Eiwit totaal (BL)_get_var", "Eiwit totaal (BL)_get_mean"])].values,
         'Ureum (BL)_get_var': df.columns[~df.columns.isin(
             ["Ureum (BL)_get_var", "Ureum (BL)_get_mean",
              "Albumine (BL)_get_var", "Albumine (BL)_get_mean",
              "Calcium (BL)_get_var", "Calcium (BL)_get_mean",
              "Chloride (BL)_get_var", "Chloride (BL)_get_mean",
              "Eiwit totaal (BL)_get_var", "Eiwit totaal (BL)_get_mean"])].values,
         'bmi': df.columns[~df.columns.isin(
             ["bmi"])].values,
         'prev_admission_icu': df.columns[~df.columns.isin(
             ["prev_admission_icu"])].values,
         'sbp': df.columns[~df.columns.isin(
             ["sbp", "dbp", "map", "urine_ml_kg_h_6hour",
              "Magnesium (BL)_get_var", "Magnesium (BL)_get_mean"])].values,
         'dbp': df.columns[~df.columns.isin(
             ["sbp", "dbp", "map", "urine_ml_kg_h_6hour",
              "Magnesium (BL)_get_var", "Magnesium (BL)_get_mean"])].values,
         'map': df.columns[~df.columns.isin(
             ["sbp", "dbp", "map", "urine_ml_kg_h_6hour",
              "Magnesium (BL)_get_var", "Magnesium (BL)_get_mean"])].values,
         'atrial_fibrillation': df.columns[~df.columns.isin(
             ["atrial_fibrillation"])].values,
         'urine_ml_kg_h_6hour': df.columns[~df.columns.isin(
             ["urine_ml_kg_h_6hour"])].values,
         'mech_vent_vt': df.columns[~df.columns.isin(
             ["mech_vent_vt"])].values,
         'mech_vent_peep': df.columns[~df.columns.isin(
             ["mech_vent_peep"])].values,
         'gender' : df.columns[~df.columns.isin(
             ["gender", "Eiwit totaal (BL)_get_var",
              "Eiwit totaal (BL)_get_mean"])].values,
         'apacheIV_postop1': df.columns[~df.columns.isin(
             ["apacheIV_postop1"])].values,
         'cvp': df.columns[~df.columns.isin(
             ["cvp"])].values,
         'mech_vent_24h': df.columns[~df.columns.isin(
             ["mech_vent_24h"])].values,
         'mech_vent_admission': df.columns[~df.columns.isin(
             ["mech_vent_admission"])].values,
         'worsened_resp_cond': df.columns[~df.columns.isin(
             ["worsened_resp_cond"])].values,
         'mech_vent_resp':  df.columns[~df.columns.isin(
             ["mech_vent_resp"])].values,
         'age': df.columns[~df.columns.isin(
             ["age"])].values,
         'apacheIV_score': df.columns[~df.columns.isin(
             ["apacheIV_score"])].values,
         'sapsII_score': df.columns[~df.columns.isin(
             ["sapsII_score"])].values,
         'hr_admission': df.columns[~df.columns.isin(
             ["hr_admission"])].values,
         'resp_rate': df.columns[~df.columns.isin(
             ["resp_rate"])].values,
         'FiO2_low': df.columns[~df.columns.isin(
             ["FiO2_low", "mech_vent_vt", "mech_vent_resp"])].values,
         'emv_score': df.columns[~df.columns.isin(
             ["emv_score"])].values,
         'vg_mi': df.columns[~df.columns.isin(
             ["vg_mi"])].values,
         'vg_dm': df.columns[~df.columns.isin(
             ["vg_dm"])].values,
         'vg_chroncvdinsuff': df.columns[~df.columns.isin(
             ["vg_chroncvdinsuff"])].values,
         'vg_copd': df.columns[~df.columns.isin(
             ["vg_copd"])].values,
         'vg_respinsuff': df.columns[~df.columns.isin(
             ["vg_respinsuff"])].values,
         'vg_chronnierinsuff': df.columns[~df.columns.isin(
             ["vg_chronnierinsuff"])].values,
         'vg_dialysis': df.columns[~df.columns.isin(
             ["vg_dialysis"])].values,
         'vg_cirrhose': df.columns[~df.columns.isin(
             ["vg_cirrhose"])].values,
         'vg_malign_meta': df.columns[~df.columns.isin(
             ["vg_malign_meta"])].values,
         'vg_malign_hema': df.columns[~df.columns.isin(
             ["vg_malign_hema"])].values,
         'vg_immun_insuff': df.columns[~df.columns.isin(
             ["vg_immun_insuff"])].values,
         'Eiwit totaal (BL)_get_mean': df.columns[~df.columns.isin(
             ["Eiwit totaal (BL)_get_var", "Eiwit totaal (BL)_get_mean"])].values,
         'Eiwit totaal (BL)_get_var': df.columns[~df.columns.isin(
             ["Eiwit totaal (BL)_get_var", "Eiwit totaal (BL)_get_mean"])].values}
    
    # Define MICE forest imputation setup for sics
    kernel_sics = mf.ImputationKernel(
        df, datasets=n_imputations, random_state=seed,
        mean_match_candidates = var_mmc,
        variable_schema = var_sch, save_all_iterations=True,
        mean_match_function = mmf.mean_match_kdtree_classification)
    
    # Define MICE forest imputation setup for mumc
    kernel_mumc = mf.ImputationKernel(
        df_mumc, datasets=n_imputations, random_state=seed,
        mean_match_candidates = var_mmc,
        variable_schema = var_sch, save_all_iterations=True,
        mean_match_function = mmf.mean_match_kdtree_classification)
    
    # Tune the lightgbm imputers
    t = time.time()
    optimal_parameters_sics, losses_sics = kernel_sics.tune_parameters(
        dataset=0, optimization_steps=n_opt_steps)
    print(f"SICS took {np.round((time.time() - t)/60, 0)} minutes to tune")
    t = time.time()
    optimal_parameters_mumc, losses_mumc = kernel_mumc.tune_parameters(
        dataset=0, optimization_steps=n_opt_steps)
    print(f"MUMC+ took {np.round((time.time() - t)/60, 0)} minutes to tune")
    
    # Run the MICE forest algorithm for n_iterations on each of the datasets
    t = time.time()
    kernel_sics.mice(n_iterations, n_jobs=-1,
                     variable_parameters=optimal_parameters_sics)
    print(f"sics took {np.round((time.time() - t)/60, 0)} minutes to impute")
    t = time.time()
    kernel_mumc.mice(n_iterations, n_jobs=-1,
                     variable_parameters=optimal_parameters_mumc)
    print(f"MUMC+ took {np.round((time.time() - t)/60, 0)} minutes to impute")
    
    # Store imputed data sets
    df = kernel_sics.complete_data(0)
    df_mumc = kernel_mumc.complete_data(0)
    
    # Plot some imputation results if desired
    if showit:
        
        # Create disitribution plots for variables with >10 imputed values
        kernel_sics.plot_imputed_distributions(wspace=0.3,hspace=0.3,
                                               variables = list(
                                                   var_mis.variable[
                                                       var_mis["count"] > 10]))
        plt.tight_layout()
        plt.savefig("figures/imputed_distributions_sics.svg")
        kernel_sics.plot_correlations()
        plt.tight_layout()
        plt.savefig("figures/imputed_correlations_sics.svg")
        plt.figure()
        kernel_sics.plot_feature_importance(0, annot=False,cmap="YlGnBu",vmin=0,
                                            vmax=0.3, linewidth=0.5,
                                            linecolor = "black")
        plt.tight_layout()
        plt.savefig("figures/imputed_feature_importance_sics.svg")
        
        # Create disitribution plots for variables with >10 imputed values
        kernel_mumc.plot_imputed_distributions(wspace=0.3,hspace=0.3,
                                               variables = list(
                                                   var_mis_mumc.variable[
                                                       var_mis_mumc["count"] > 10]))
        plt.tight_layout()
        plt.savefig("figures/imputed_distributions_mumc.svg")
        kernel_mumc.plot_correlations()
        plt.tight_layout()
        plt.savefig("figures/imputed_correlations_mumc.svg")
        plt.figure()
        kernel_mumc.plot_feature_importance(0, annot=False,cmap="YlGnBu",vmin=0,
                                            vmax=0.3, linewidth=0.5,
                                            linecolor = "black")
        plt.tight_layout()
        plt.savefig("figures/imputed_feature_importance_mumc.svg")
        
    return df, df_mumc, kernel_sics, kernel_mumc


def impute_data_original_sics(mat, seed = 5192):
    index_names = mat.index
    col_names = mat.columns
    
    imp = IterativeImputer(max_iter=10, random_state=seed)
    mat = pd.DataFrame(imp.fit_transform(mat), index = index_names,
                       columns = col_names)
    
    return(mat)