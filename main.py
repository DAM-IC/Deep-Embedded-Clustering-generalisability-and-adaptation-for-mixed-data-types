# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:06:48 2022

@author: Jip de Kok

This is the main script which runs all individual steps from loading the data
to clustering and visualising the results. It should be run in its entirety.
"""
# %% preparing the workspace
# Import packages (these should be installed beforehand)
import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import keras
import time
import warnings

# Set working directory to current file directory
# Uncomment the next line and specify path if not using spyder project or are
# encountering other issues with file/folder directories.
#os.chdir("PATH/TO/FOLDER/CONTAINING/main.py")

# Add scripts folder to sys.path so scripts can be imported
sys.path.insert(1, 'scripts')

# Ensure proper folder structure
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('stats'):
    os.makedirs('stats')
    
# Import functions from local scripts
from data_functions import load_sics_data, compute_missingness
from data_functions import load_mumc_data, scale_data
from clustering_functions import compute_cluster_stability, compare_clusters
from clustering_functions import sample_cluster_stability, jaccard_similarity
from visualisation import cluster_heatmap, compare_variables, cluster_pca
from visualisation import pca_score_plot, cluster_heatmap_mumc
from visualisation import plot_missingness_pattern, plot_stability
from imputation import impute_data
from xvae import load_xvae_model

# Set default plot setting
sns.set(rc={'figure.figsize':(11.7,8.27)},
        font_scale = 1.5)
sns.set_palette(sns.color_palette("deep"))
sns.set_style("whitegrid")

# Set seed number
seed = 5192

# Specify folder location to save data to (and potenitally load it form)
datafolder = "data/"

# Specify if files and figures should be saved locally (will overwrite)
save_file = True

# %% Loadprevious model results
load_previous_data = True # Set to False if you want to load your own data
load_previous_results = False # Set to True if you want to load previous clustering results

# Load already pre-processed data
if load_previous_data:
    # Load the data
    df = pd.read_csv(f"{datafolder}df_dummy_data.csv", index_col = 0)
    df_mumc = pd.read_csv(f"{datafolder}df_mumc_dummy_data.csv", index_col = 0)
    descriptives = pd.read_csv(f"{datafolder}descriptives_dummy_data.csv", index_col = 0)
    descriptives_mumc = pd.read_csv(f"{datafolder}descriptives_mumc_dummy_data.csv", index_col = 0)
    # Set appropriate dtypes
    cat_vars = ['worsened_resp_cond', 'gender', 'apacheIV_postop1', 'cvp',
                'mech_vent_admission', 'mech_vent_24h', "atrial_fibrillation",
                "prev_admission_icu", "vg_mi", "vg_dm", "vg_chroncvdinsuff",
                "vg_copd", "vg_respinsuff", "vg_chronnierinsuff", "vg_dialysis",
                "vg_cirrhose", "vg_malign_meta", "vg_malign_hema", "vg_immun_insuff"]
    df[cat_vars] = df[cat_vars].astype("category")
    df_mumc[cat_vars] = df_mumc[cat_vars].astype("category")
    

if load_previous_results:
    # Original DEC model
    y_pred_sics_orig = pd.read_csv(f"{datafolder}y_pred_sics_orig.csv",
                                   index_col = None, header = None)
    y_pred_mumc_orig = pd.read_csv(f"{datafolder}y_pred_mumc_orig.csv", header= None)
    cluster_stab_sics_orig = pd.read_csv(f"{datafolder}sics_orig/"\
                                  "cluster_stab_sics_orig.csv",
                                  index_col = 0)
    Z_sics_orig = pd.read_csv(f"{datafolder}sics_orig/Z_sics_orig.csv",
                              index_col = 0)
    Z_mumc_orig = pd.read_csv(f"{datafolder}sics_orig/Z_mumc_orig.csv",
                              index_col = 0)
    centroids_sics_orig = pd.read_csv(f"{datafolder}sics_orig/"\
                                      "centroids_sics_orig.csv",
                                      index_col = 0)
    y_pred_stab_sics_orig = pd.read_csv(f"{datafolder}sics_orig/y_pred_stab_sics"\
                                        "_orig.csv", index_col = 0)

    model_sics_orig = keras.models.load_model(f"{datafolder}sics_orig/model_sics_orig")
    encoder_sics_orig = keras.models.load_model(f"{datafolder}sics_orig/encoder_sics_orig")


    #X-DEC model
    y_pred_mumc_xvae = pd.read_csv(f"{datafolder}y_pred_mumc_xvae.csv",
                                          header = None)
    cluster_stab_sics_xvae = pd.read_csv(f"{datafolder}sics_xvae/"\
                                  "cluster_stab_sics_xvae.csv", index_col = 0)
    Z_sics_xvae = pd.read_csv(f"{datafolder}sics_xvae/Z_sics_xvae.csv",
                              index_col = 0)
    y_pred_sics_xvae = pd.read_csv(f"{datafolder}sics_xvae/"\
                                   "y_pred_sics_xvae.csv",
                                   index_col = 0)
    centroids_sics_xvae = pd.read_csv(f"{datafolder}sics_xvae/"\
                                      "centroids_sics_xvae.csv", index_col= 0)
    y_pred_stab_sics_xvae = pd.read_csv(f"{datafolder}sics_xvae/y_pred_stab_sics_xvae.csv",
                                        index_col = 0)
    model_sics_xvae = keras.models.load_model(f"{datafolder}sics_xvae/model_sics_xvae")
    encoder_sics_xvae = load_xvae_model(f"{datafolder}sics_xvae/encoder_sics_xvae2")
        

# Load and pre-process raw data - this will not work as without the source data
if load_previous_data == False:
        # Load the SICS data
        df, y_mort, y_aki, var_arr, descriptives = load_sics_data(
            scale_data = False, binary_cvp = True,
            time_filter = None)

        # Load the MUMC+ data
        (df_mumc, y_mort_mumc, y_aki_mumc,
         var_arr_mumc, descriptives_mumc) = load_mumc_data(scale_data = False,
                                                           sample_selection = False,
                                                           time_filter = None,
                                                           medium_care = False)

        # Only keep variables present in both data sets
        df = df.loc[:,np.isin(df.columns, df_mumc.columns)]
        df_mumc = df_mumc.loc[:,np.isin(df_mumc.columns, df.columns)]

        # Make sure variables are in same order
        df_mumc = df_mumc[df.columns]

        # Save data locally
        if save_file:
            df.to_csv(f"{datafolder}df_sics.csv")
            df_mumc.to_csv(f"{datafolder}df_mumc.csv")

# %% Imputation
if load_previous_results == False:
        # Compute degree of missingness
        var_mis, sam_mis = compute_missingness(
            df, showit = True, threshold = 0.01, save = save_file,
            savename="degrees of missingness SICS")
        var_mis_mumc, sam_mis_mumc = compute_missingness(
            df_mumc, showit = True, save = save_file,
            savename="degrees of missingness MUMC+", threshold = 0.01)

        # Plot the misingness pattern for the MUMC+ data
        plot_missingness_pattern(df_mumc)

        # Extract column datatypes
        dtypes = df.dtypes

        # Impute the missing data
        df, df_mumc, kernel_sics, kernel_mumc = impute_data(df, df_mumc,
                                                            showit = True,
                                                            n_imputations = 5,
                                                            n_opt_steps=10,
                                                            n_iterations=10,
                                                            seed = seed)

        df = df.astype(dtypes)
        df_mumc = df_mumc.astype(dtypes)

        # Save imputation results locally
        if save_file:
            kernel_sics.save_kernel(f"{datafolder}sics_kernel")
            kernel_mumc.save_kernel(f"{datafolder}mumc_kernel")
            df.to_csv(f"{datafolder}df_sics_dummy_imputed.csv")
            df_mumc.to_csv(f"{datafolder}df_mumc_dummy_imputed.csv")


# %% Scale data
if load_previous_results == False:
    scale = "standardise"
    if scale == "LogMinMax":
        df_index = df.index.values
        df_mumc_index = df_mumc.index.values
        # Log scale numerical variables
        df, df_scaler = scale_data(df, method = "log")
        # Min-max scale the log transformed numeric variables and save scaler
        df, df_scaler = scale_data(df, method = "minmax")
        # Log scale numerical variables
        df_mumc, df_mumc_scaler = scale_data(df_mumc, method = "log")
        # Apply the saved min-max scaler on the log transformed numeric variables
        df_mumc= pd.DataFrame(df_scaler.transform(df_mumc),
                              columns = df_mumc.columns)
        df.index = df_index
        df_mumc.index = df_mumc_index
    elif scale == "robustscaler":
        df, scaler = scale_data(df, method = scale)
        
        cat_features = df_mumc.columns[df_mumc.dtypes == "category"]
        df_numeric = df_mumc.drop(cat_features, axis = 1)
        df_numeric = pd.DataFrame(scaler.transform(
            df_numeric), columns = df_numeric.columns)
        # Combine scaled numeric features with unscaled categorical features
        df_mumc = pd.concat([df_mumc.reset_index()[cat_features], df_numeric],
                       axis = 1).set_index(df_mumc.index)
    elif scale == "standardise":
        df, scaler = scale_data(df, method = scale)
        cat_features = df_mumc.columns[df_mumc.dtypes == "category"]
        df_numeric = df_mumc.drop(cat_features, axis = 1)
        df_numeric = pd.DataFrame(scaler.transform(
            df_numeric), columns = df_numeric.columns)
        # Combine scaled numeric features with unscaled categorical features
        df_mumc = pd.concat([df_mumc.reset_index()[cat_features], df_numeric],
                       axis = 1).set_index(df_mumc.index)
    
    # Save all data locally
    if save_file:
        df.to_csv(f"{datafolder}df_sics_imputed_scaled.csv")
        df_mumc.to_csv(f"{datafolder}df_mumc_imputed_scaled.csv")
        descriptives.to_csv(f"{datafolder}descriptives_sics.csv")
        descriptives_mumc.to_csv(f"{datafolder}descriptives_mumc.csv")
    
# %% Explorative data analysis

# Compare variable distributions
compare_variables(df, df_mumc, plot_type = "violinplot", save = save_file,
                  seperator = ["mean", "var", "other"])

# PCA
pca_score_plot(df, df_mumc, scale = False, plot_3D = False, biplot = False,
               multi_plot=True, save = save_file,
               dataset_names = ["SICS", "MUMC+"], threshold = 0.25,
               components = [1,2,3], filetype="png")


pca_score_plot(df, df_mumc, scale = False, plot_3D = True, biplot = False,
               dataset_names = ["SICS", "MUMC+"], threshold = 0.25,
               components = [1,2,3], save =save_file, filetype="png")

# %% The recreated DEC model

# Get data as numpy arrays
x = np.asarray(df)
x_mumc = np.asarray(df_mumc)

# Specify some DEC parameters
n_cluster = 6 # Number of clusters
batch_size = 64
e = 8 # Number of neurons in the encoding layer
h = 64 # Number of neurons in the hidden layer
k = 10 # how many splits should be made, each split will have k percentage of samples removed
rep = 100 # How often to split the data. Stability will be computed on k*rep subsets.

# Run original SICS clustering model and compute cluster stability
t_orig = time.time()
(cluster_stab_sics_orig, Z_sics_orig, y_pred_sics_orig,
centroids_sics_orig, y_pred_stab_sics_orig, model_sics_orig,
 encoder_sics_orig) = compute_cluster_stability(x, n_cluster = 6,
     k = k, rep = rep, mapper = "jaccard", majority_vote=True, save = save_file,
     return_extra = True,
     save_name=f"Orginal_SICS_model_clustering_stabiltity_k{k}_rep_{rep}")
t_orig = time.time() - t_orig

# Compute Jaccard cluster stability
y_pred_stab_sics_orig, cluster_stab_sics_orig = jaccard_similarity(
    y_pred_sics_orig, y_pred_stab_sics_orig, True)
# Compute samplewise cluster stability
sample_sics_orig = sample_cluster_stability(True, y_pred_sics_orig,
                                       y_pred_stab_sics_orig)

# Create stability plots
plot_stability(cluster_stab_sics_orig, stability_type = "cluster", save = save_file,
               savename = "Jaccard stability boxplot - majority vote - original SICS model")
plot_stability(sample_sics_orig, stability_type = "sample", save = save_file,
               mutation = 0.2,
               savename = "Sample stability - majority vote - original SICS model")

# Apply model to MUMC+ data
y_pred_mumc_orig = model_sics_orig.predict(x_mumc).argmax(1)
Z_mumc_orig = encoder_sics_orig.predict(x_mumc)

# Visualise phenotype heatmaps
df_viz_sics_orig = descriptives.copy().reset_index(drop=True)
df_viz_sics_orig["cluster"] = y_pred_sics_orig + 1

# Create heatmap of outcomes and clinical-endpoints per cluster
# SICS_orig
cluster_heatmap(df_viz_sics_orig, save = save_file, method = "default",
                outcome = "admission", diagnosis_focus = True, annot = True,
                title = "Recreated DEC - SICS",
                savename = "heatmap of admission outcomes sics_original"\
                    " annotated - default")
    
# PCA score plots
# On latent feature space
cluster_pca(Z_sics_orig, y_pred_sics_orig+1, plot_3D = False, multi_plot = True,
            title = "DEC - SICS",
            save = save_file, savename = "Z Cluster PCA sics_original")
# On original input feature space
cluster_pca(x, y_pred_sics_orig+1, plot_3D = False, multi_plot = True,
            save = save_file, savename = "x Cluster PCA sics_original")

# MUMC_orig
df_viz_mumc_orig = descriptives_mumc.copy().reset_index(drop=True)
df_viz_mumc_orig["cluster"] = y_pred_mumc_orig + 1

cluster_heatmap_mumc(df_viz_mumc_orig, save = save_file, method = "default",
                     diagnosis_focus = True, annot = True,
                     title = "Recreated DEC - MUMC+",
                     savename = "heatmap of admission outcomes mumc_original"\
                         " annotated - default")


# Create cluster plots
cluster_pca(Z_mumc_orig, y_pred_mumc_orig+1, plot_3D = False, multi_plot = True,
            title = "DEC - MUMC+", 
            save = save_file, savename = "Z Cluster PCA mumc_original")
cluster_pca(x_mumc, y_pred_mumc_orig+1, plot_3D = False, multi_plot = True,
            save = save_file, savename = "x Cluster PCA mumc_original")

# Check if clusters are the same across hospitals
sics_orig_cluster_mappings = compare_clusters(y_pred_sics_orig,
                                              y_pred_mumc_orig,
                                              Z_sics_orig, Z_mumc_orig,
                                              df, df_mumc,
                                              descriptives, descriptives_mumc,
                                              exclusive_mapping=False,
                                              save = save_file, filetype = "svg",
                                              savename = "sics_orig_cluster_mappings")

# Save the results locally
if save_file:
    if not os.path.exists(f"{datafolder}sics_orig"):
        os.makedirs(f"{datafolder}sics_orig")
    pd.DataFrame(y_pred_sics_orig).to_csv(f"{datafolder}y_pred_sics_orig.csv",
                                          header =False, index = False)
    pd.DataFrame(y_pred_mumc_orig).to_csv(f"{datafolder}y_pred_mumc_orig.csv",
                                          header = False, index = False)
    cluster_stab_sics_orig.to_csv(f"{datafolder}sics_orig/"\
                                  "cluster_stab_sics_orig.csv")
    pd.DataFrame(Z_sics_orig).to_csv(f"{datafolder}sics_orig/Z_sics_orig.csv")
    pd.DataFrame(y_pred_sics_orig).to_csv(f"{datafolder}sics_orig/"\
                                          "y_pred_sics_orig.csv")
    pd.DataFrame(centroids_sics_orig).to_csv(f"{datafolder}sics_orig/"\
                                             "centroids_sics_orig.csv")
    y_pred_stab_sics_orig.to_csv(f"{datafolder}sics_orig/y_pred_stab_sics_orig.csv")
    model_sics_orig.save(f"{datafolder}sics_orig/model_sics_orig")
    encoder_sics_orig.save(f"{datafolder}sics_orig/encoder_sics_orig")


# %% Tune X-DEC
# separate categorical from numerical variables
cat_features = df.columns[df.dtypes == "category"]
num_features = df.columns[df.dtypes != "category"]
x_cat = np.array(df[cat_features])
x_num = np.array(df[num_features])
feature_array = [x_num, x_cat]

# Define (hyper)parameter values of of X-DEC to evaluate
k=10 # Number of folds in repeated k-fold cross validation
rep = 3 # Number of repeats in repeated k-fold cross validation
n_cluster_list = [6] # Number of clusters
mapper = "jaccard" # Mapper to determine how clusters are mapped to each other.
ds1_list = [26, 36, 50] # The number of neurons in the first hidden layer of input set 1
ds2_list = [15, 10, 5] # The number of neurons in the first hidden layer of input set 2
ds12_list = [30, 38 , 46, 60] # The number of neurons in the hidden layer that joins input set 1 and 2
ls_list = [15, 20, 25, 30] # The number of neurons in the embedding layer
act_list = ["elu"] # The activation function
dropout_list = [0.1] # Dropout
distance_list = ["mmd"] # Distance metric to use for regularisation in the objective function in xvae
epochs_list = [250] # Number of epochs
bs_list = [64] # Batch size
beta_list = [200] # Beta (the influence of the disentanglement factor)

# Generate hyperparameter grid
hyperparameter_grid = np.array([(n_cluster, ds1, ds2, ds12, ls, act, dropout, distance,
                                 epochs, bs, beta) for n_cluster in n_cluster_list for ds1 in ds1_list for ds2 
                                in ds2_list for ds12 in ds12_list for ls in 
                                ls_list for act in act_list for dropout in 
                                dropout_list for distance in distance_list for 
                                epochs in epochs_list for bs in bs_list for 
                                beta in beta_list])
result_df = pd.DataFrame(hyperparameter_grid, columns = ['n_cluster', 'ds1','ds2','ds12',
                         'ls','act','dropout','distance','epochs','bs','beta'])
result_df['stability_mean'] = np.nan
result_df['stability_stdev'] = np.nan

# Initiate temp variable to store stability scores
df_stab_temp = np.repeat(np.nan, k*rep)

# Run X-DEC and compute stability for all hyperparameter cominations
t_xvae_tuning = time.time()
warnings.filterwarnings('ignore')
for i in range(len(hyperparameter_grid)):
    try:
        (cluster_stab, Z, labels, centroids, y_pred,
        model, encoder) = compute_cluster_stability(
            feature_array,
            n_cluster = hyperparameter_grid[i][0].astype(int),
            model_type= "vae",
            k = k, rep = rep,
            ds1 = hyperparameter_grid[i][1].astype(int),
            ds2 = hyperparameter_grid[i][2].astype(int),
            ds12 = hyperparameter_grid[i][3].astype(int),
            ls = hyperparameter_grid[i][4].astype(int),
            act = hyperparameter_grid[i][5],
            dropout = hyperparameter_grid[i][6].astype(float),
            distance = hyperparameter_grid[i][7],
            epochs = hyperparameter_grid[i][8].astype(int),
            batch_size = hyperparameter_grid[i][9].astype(int),
            beta = hyperparameter_grid[i][10].astype(int),
            retrain_autoencoder = True,
            majority_vote=True,
            seed=seed, mapper = mapper,
            return_extra = True, save = save_file,
            save_name = f"SICS_xvae_hyperparameter_set_{i}")
        if mapper == "jaccard":
            result_df.loc[i,"stability_mean"] = cluster_stab.mean(axis=1).mean()
            result_df.loc[i,"stability_stdev"] = cluster_stab.std(axis=1).mean()
        else:
            result_df.loc[i,"stability_mean"] = np.mean(cluster_stab)
        
    except:
        print("Error: stability could not be computed."\
              "\nSkipping to next iteration.")
    print(f"stability: {round(np.mean(cluster_stab),2)}%")
    print(f"Completed {(i+1)}/{len(hyperparameter_grid)} clustering architectures.")
    print(f"Average iteration time: {np.round(((time.time()-t_xvae_tuning)/(i+1))/60,2)} minutes")
    
    # Save results every 10 iterations
    if i%10 == 0:
        if save_file:
            result_df.to_csv("stats/SICS_xvae_clustering_optimisation_results_temp.csv",
                             index = False)

print(f"Cluster tuning took {np.round((time.time()-t_xvae_tuning)/60,0)} minutes")

# Save hyperparameter optimisation results locally
if save_file:
    result_df.to_csv("stats/SICS_xvae_clustering_optimisation_results_17-02-2023_mumc_30it.csv",
                     index = False)

t_xvae_tuning = time.time() - t_xvae_tuning

# Extract optimal hyperparameters
optimal_hyperparameters = result_df[result_df.stability_mean == np.max(
    result_df.stability_mean)]

# Save the results locally
if save_file:
    if not os.path.exists(f"{datafolder}SICS_xvae_tuned"):
        os.makedirs(f"{datafolder}SICS_xvae_tuned")
    pd.DataFrame(labels).to_csv(f"{datafolder}y_pred_SICS_xvae_tuned.csv",
                                          header =False, index = False)
    cluster_stab.to_csv(f"{datafolder}SICS_xvae_tuned/"\
                        "cluster_stab_mumc_xvae_tuned.csv")
    pd.DataFrame(Z).to_csv(f"{datafolder}SICS_xvae_tuned/Z_sics_xvae_tuned.csv")
    pd.DataFrame(centroids).to_csv(f"{datafolder}SICS_xvae_tuned/"\
                                             "centroids_SICS_xvae_tuned.csv")
    y_pred.to_csv(f"{datafolder}SICS_xvae_tuned/y_pred_stab_SICS_xvae_tuned.csv")
    model.save(f"{datafolder}SICS_xvae_tuned/model_SICS_xvae_tuned")
    encoder.save_model(f"{datafolder}SICS_xvae_tuned/encoder_sics_xvae_tuned")

# %% Validate X-DEC model

# specify X-DEC parameters
k = 10
rep = 100
n_cluster = 6
mapper = "jaccard"
act = "elu"
dropout = 0.1
distance = "mmd"
epochs = 250
bs = 64
beta = 200

# Esnure some paramteters are integers
ds1 = int(optimal_hyperparameters.ds1)
ds2 = int(optimal_hyperparameters.ds2)
ds12 = int(optimal_hyperparameters.ds12)
ls = int(optimal_hyperparameters.ls)

# Run X-DEC and compute cluster stability
t_xvae = time.time()
(cluster_stab_sics_xvae, Z_sics_xvae, y_pred_sics_xvae,
 centroids_sics_xvae, y_pred_stab_sics_xvae, model_sics_xvae,
 encoder_sics_xvae) = compute_cluster_stability(
    feature_array, n_cluster,
    model_type= "vae",
    k = k, rep = rep,
    ds1 = ds1,
    ds2 = ds2,
    ds12 = ds12,
    ls = ls,
    act = act,
    dropout = dropout,
    distance = distance,
    epochs = epochs,
    batch_size = bs,
    beta = beta,
    retrain_autoencoder = True,
    seed=seed,
    mapper = mapper,
    weighted = True,
    majority_vote= True,
    disable_resampling = False,
    return_extra = True, save = save_file,
    save_name = "XVAE_SICS_model_clustering_stabiltity_k10_rep_100_23-02-2023")
t_xvae = time.time()-t_xvae # elapsed time in seconds

# Compute Jaccard cluster stability
y_pred_stab_sics_xvae, cluster_stab_sics_xvae = jaccard_similarity(
    y_pred_sics_xvae, y_pred_stab_sics_xvae, True)
# Compute samplewise cluster stability
sample_sics_xvae = sample_cluster_stability(True, y_pred_sics_xvae,
                                       y_pred_stab_sics_xvae)
plot_stability(cluster_stab_sics_xvae, stability_type = "cluster", save = save_file,
               savename = "Jaccard stability boxplot - majority vote - xvae SICS model")
plot_stability(sample_sics_xvae, stability_type = "sample", save = save_file,
               mutation = 0.5,
               savename = "Sample stability - majority vote - xvae SICS model")

# Generate PCA plots of clustering results
cluster_pca(Z_sics_xvae, y_pred_sics_xvae+1, plot_3D = False, multi_plot = True,
            title = "X-DEC - SICS",
            save = save_file, savename = "Z Cluster PCA sics_xvae")
cluster_pca(x, y_pred_sics_xvae+1, plot_3D = False, multi_plot = True,
            save = save_file, savename = "x Cluster PCA sics_xvae")
cluster_pca(Z_sics_xvae, y_pred_sics_xvae+1, plot_3D = False, multi_plot = False,
            stability = sample_sics_xvae, title = "original feature space - SICS",
            save = save_file, savename = "x Cluster PCA sics_xvae")

# Apply model on MUMC+ data
x_cat_mumc = np.array(df_mumc[cat_features])
x_num_mumc = np.array(df_mumc[num_features])
feature_array_mumc = [x_num_mumc, x_cat_mumc]
y_pred_mumc_xvae = model_sics_xvae.predict(feature_array_mumc).argmax(1)
Z_mumc_xvae = encoder_sics_xvae.predict(x_num_mumc, x_cat_mumc)

# Visualise phenotype heatmaps
df_viz_sics_xvae = descriptives.copy().reset_index(drop=True)
df_viz_sics_xvae["cluster"] = y_pred_sics_xvae + 1

# Create heatmap of outcomes and clinical-endpoints per cluster
# SICS_xvae
cluster_heatmap(df_viz_sics_xvae, save = save_file, method = "default",
                outcome = "admission", diagnosis_focus = True, annot = True,
                title = "X-DEC - SICS", vmax = 84/200,
                savename = "heatmap of admission outcomes sics_xvae"\
                    " annotated - default")

# MUMC_xvae
df_viz_mumc_xvae = descriptives_mumc.copy().reset_index(drop=True)
df_viz_mumc_xvae["cluster"] = y_pred_mumc_xvae + 1

cluster_heatmap_mumc(df_viz_mumc_xvae, save = save_file, method = "default",
                     diagnosis_focus = True, annot = True,
                     title = "X-DEC - MUMC+",
                     savename = "heatmap of admission outcomes mumc_xvae"\
                         " annotated - default", vmax = 84/200)
    
cluster_pca(Z_mumc_xvae, y_pred_mumc_xvae+1, plot_3D = True, multi_plot = True,
            title = "X-DEC - MUMC+", 
            save = save_file, savename = "Z Cluster PCA mumc_xvae")
cluster_pca(df_mumc, y_pred_mumc_xvae, plot_3D = False, multi_plot = True,
            title = "original feature space - MUMC+",
            save = save_file, savename = "x Cluster PCA mumc_xvae")

xvae_cluster_mappings = compare_clusters(y_pred_sics_xvae,
                                         y_pred_mumc_xvae,
                                         Z_sics_xvae, Z_mumc_xvae,
                                         df, df_mumc,
                                         descriptives,
                                         descriptives_mumc,
                                         exclusive_mapping= False,
                                         save = save_file, filetype = "svg",
                                         savename = "sics_xvae_cluster_mappings")

# Save the results locally
if save_file:
    if not os.path.exists(f"{datafolder}sics_xvae"):
        os.makedirs(f"{datafolder}sics_xvae")
    pd.DataFrame(y_pred_sics_xvae).to_csv(f"{datafolder}y_pred_sics_xvae.csv",
                                          header =False, index = False)
    pd.DataFrame(y_pred_mumc_xvae).to_csv(f"{datafolder}y_pred_mumc_xvae.csv",
                                          header = False, index = False)
    cluster_stab_sics_xvae.to_csv(f"{datafolder}sics_xvae/"\
                                  "cluster_stab_sics_xvae.csv")
    pd.DataFrame(Z_sics_xvae).to_csv(f"{datafolder}sics_xvae/Z_sics_xvae.csv")
    pd.DataFrame(y_pred_sics_xvae).to_csv(f"{datafolder}sics_xvae/"\
                                          "y_pred_sics_xvae.csv")
    pd.DataFrame(centroids_sics_xvae).to_csv(f"{datafolder}sics_xvae/"\
                                             "centroids_sics_xvae.csv")
    y_pred_stab_sics_xvae.to_csv(f"{datafolder}sics_xvae/y_pred_stab_sics_xvae.csv")
    model_sics_xvae.save(f"{datafolder}sics_xvae/model_sics_xvae")
    encoder_sics_xvae.save_model(f"{datafolder}sics_xvae/encoder_sics_xvae")


# %% Tune DEC on SICS data
# Get data as numpy arrays
x = np.asarray(df)
x_mumc = np.asarray(df_mumc)

# Specify some DEC parameters
n_cluster_list = [6]
batch_size_list = [64]
e_list= [4, 8, 12, 16]
h_list = [16, 32, 64, 128]
mapper = "jaccard"
k = 10
rep = 3

# Generate hyperparameter grid
hyperparameter_grid_dec = np.array([(n_cluster, batch_size, e, h) 
                                for n_cluster in n_cluster_list
                                for batch_size in batch_size_list
                                for e in e_list
                                for h in h_list])
result_df_dec = pd.DataFrame(hyperparameter_grid_dec, columns = ['n_cluster',
                                                         'batch_size',
                                                         'e',
                                                         'h'])
result_df_dec['stability_mean'] = np.nan
result_df_dec['stability_stdev'] = np.nan

# Initiate temp variable to store stability scores
df_stab_temp_dec = np.repeat(np.nan, k*rep)

# Run X-DEC and compute stability for all hyperparameter cominations
t_dec_tuning = time.time()
warnings.filterwarnings('ignore')
for i in range(len(hyperparameter_grid_dec)):
    try:
        (cluster_stab_dec, Z_dec, y_pred_dec,
        centroids_dec, y_pred_stab_dec, model_dec,
         encoder_dec) = compute_cluster_stability(
             x, n_cluster = hyperparameter_grid_dec[i][0].astype(int),
             batch_size = hyperparameter_grid_dec[i][1].astype(int),
             neurons_e = hyperparameter_grid_dec[i][2].astype(int),
             neurons_h = hyperparameter_grid_dec[i][3].astype(int),
             k = k, rep = rep, mapper = mapper, majority_vote=True,
             save = save_file,
             return_extra = True,
             save_name=f"Orginal_SICS_model_hyperparamter_set{i}")
        if mapper == "jaccard":
            result_df_dec.loc[i,"stability_mean"] = cluster_stab_dec.mean(axis=1).mean()
            result_df_dec.loc[i,"stability_stdev"] = cluster_stab_dec.std(axis=1).mean()
        else:
            result_df_dec.loc[i,"stability_mean"] = np.mean(cluster_stab_dec)
    except:
        print("Error: stability could not be computed."\
              "\nSkipping to next iteration.")
    print(f"stability: {round(np.mean(cluster_stab_dec),2)}%")
    print(f"Completed {(i+1)}/{len(hyperparameter_grid_dec)} clustering architectures.")
    print(f"Average iteration time: {np.round(((time.time()-t_dec_tuning)/(i+1))/60,2)} minutes")
    
    # Save results every 10 iterations
    if i%10 == 0:
        result_df_dec.to_csv("stats/SICS_dec_clustering_optimisation_results_temp.csv",
                         index = False)

print(f"Cluster tuning took {np.round((time.time()-t_dec_tuning)/60,0)} minutes")

# Save hyperparameter optimisation results locally
result_df_dec.to_csv("stats/SICS_dec_clustering_optimisation_results_03-11-2023_mumc_30it.csv",
                     index = False)

t_dec_tuning = time.time() - t_dec_tuning

# Extract optimal hyperparameters
optimal_hyperparameters_dec = result_df_dec[result_df_dec.stability_mean == np.max(
    result_df_dec.stability_mean)]

# Save the results locally
if save_file:
    if not os.path.exists(f"{datafolder}SICS_dec_tuned"):
        os.makedirs(f"{datafolder}SICS_dec_tuned")
    pd.DataFrame(y_pred_dec).to_csv(f"{datafolder}y_pred_SICS_dec_tuned.csv",
                                    header =False, index = False)
    cluster_stab_dec.to_csv(f"{datafolder}SICS_dec_tuned/"\
                        "cluster_stab_mumc_dec_tuned.csv")
    pd.DataFrame(Z_dec).to_csv(f"{datafolder}SICS_dec_tuned/Z_sics_dec_tuned.csv")
    pd.DataFrame(centroids_dec).to_csv(f"{datafolder}SICS_dec_tuned/"\
                                             "centroids_SICS_dec_tuned.csv")
    y_pred_stab_dec.to_csv(f"{datafolder}SICS_dec_tuned/y_pred_stab_SICS_dec_tuned.csv")
    model_dec.save(f"{datafolder}SICS_dec_tuned/model_SICS_dec_tuned")
    encoder_dec.save(f"{datafolder}SICS_dec_tuned/encoder_sics_dec_tuned")


# %% Validate recreated SICS model with optimised hyper parameter values
# Specify if files and figures should be saved locally (will overwrite)
# Get data as numpy arrays
x = np.asarray(df)
x_mumc = np.asarray(df_mumc)

# Specify some DEC parameters
n_cluster = 6
batch_size = 64
e_tuned  = int(optimal_hyperparameters_dec.e)
h_tuned  = int(optimal_hyperparameters_dec.h)
k = 10
rep = 100

# Run original SICS clustering model and compute cluster stability
t_orig_tuned = time.time()
(cluster_stab_sics_orig_tuned , Z_sics_orig_tuned , y_pred_sics_orig_tuned ,
centroids_sics_orig_tuned , y_pred_stab_sics_orig_tuned , model_sics_orig_tuned ,
 encoder_sics_orig_tuned ) = compute_cluster_stability(x, n_cluster = 6,
     k = k, rep = rep, mapper = "jaccard", majority_vote=True, save = save_file,
     neurons_e = e_tuned, neurons_h = h_tuned,
     return_extra = True,
     save_name=f"Orginal_SICS_model_tuned _clustering_stabiltity_k{k}_rep_{rep}")
t_orig_tuned = time.time() - t_orig_tuned

# Compute Jaccard cluster stability
y_pred_stab_sics_orig_tuned , cluster_stab_sics_orig_tuned  = jaccard_similarity(
    y_pred_sics_orig_tuned , y_pred_stab_sics_orig_tuned , True)
# Compute samplewise cluster stability
sample_sics_orig_tuned  = sample_cluster_stability(True, y_pred_sics_orig_tuned ,
                                       y_pred_stab_sics_orig_tuned )

# Create stability plots
plot_stability(cluster_stab_sics_orig_tuned , stability_type = "cluster", save = save_file,
               savename = "Jaccard stability boxplot - majority vote - original SICS model tuned")
plot_stability(sample_sics_orig_tuned, stability_type = "sample", save = save_file,
               mutation = 0.2,
               savename = "Sample stability - majority vote - original SICS model tuned")

# Apply model to MUMC+ data
y_pred_mumc_orig_tuned  = model_sics_orig_tuned .predict(x_mumc).argmax(1)
Z_mumc_orig_tuned  = encoder_sics_orig_tuned .predict(x_mumc)

# Visualise phenotype heatmaps
df_viz_sics_orig_tuned  = descriptives.copy().reset_index(drop=True)
df_viz_sics_orig_tuned ["cluster"] = y_pred_sics_orig_tuned  + 1

# Create heatmap of outcomes and clinical-endpoints per cluster
# SICS_orig
cluster_heatmap(df_viz_sics_orig_tuned , save = save_file, method = "default",
                outcome = "admission", diagnosis_focus = True, annot = True,
                title = "Recreated DEC tuned - SICS",
                savename = "heatmap of admission outcomes sics_original_tuned "\
                    " annotated - default", vmax = 144/241) # VMAX IS FIXATED HERE!
cluster_heatmap(df_viz_sics_orig_tuned, save = save_file, method = "other",
                outcome = "admission", diagnosis_focus = True, annot = True,
                savename = "heatmap of admission outcomes sics_original_tuned "\
                    " annotated - other")

cluster_pca(Z_sics_orig_tuned, y_pred_sics_orig_tuned+1, plot_3D = False, multi_plot = True,
            title = "DEC tuned - SICS",
            save = save_file, savename = "Z Cluster PCA sics_original_tuned ")
cluster_pca(x, y_pred_sics_orig_tuned+1, plot_3D = False, multi_plot = True,
            save = save_file, savename = "x Cluster PCA sics_original_tuned ")

# MUMC_orig
df_viz_mumc_orig_tuned  = descriptives_mumc.copy().reset_index(drop=True)
df_viz_mumc_orig_tuned ["cluster"] = y_pred_mumc_orig_tuned  + 1

cluster_heatmap_mumc(df_viz_mumc_orig_tuned , save = save_file, method = "default",
                     diagnosis_focus = True, annot = True,
                     title = "Recreated DEC tuned - MUMC+",
                     savename = "heatmap of admission outcomes mumc_original_tuned "\
                         " annotated - default", vmax = 144/241) #VMAX IS FIXATED HERE!
cluster_heatmap_mumc(df_viz_mumc_orig_tuned , save = save_file, method = "other",
                     diagnosis_focus = True, annot = True,
                     savename = "heatmap of admission outcomes mumc_original_tuned "\
                         " annotated - other")

# Create cluster plots
cluster_pca(Z_mumc_orig_tuned , y_pred_mumc_orig_tuned +1, plot_3D = False, multi_plot = True,
            title = "DEC - MUMC+", 
            save = save_file, savename = "Z Cluster PCA mumc_original_tuned ")
cluster_pca(x_mumc, y_pred_mumc_orig_tuned+1, plot_3D = False, multi_plot = True,
            save = save_file, savename = "x Cluster PCA mumc_original_tuned ")

# Check if clusters are the same across hospitals
sics_orig_cluster_mappings_tuned = compare_clusters(y_pred_sics_orig_tuned,
                                              y_pred_mumc_orig_tuned,
                                              Z_sics_orig_tuned, Z_mumc_orig_tuned ,
                                              df, df_mumc,
                                              descriptives, descriptives_mumc,
                                              exclusive_mapping=False,
                                              save = save_file, filetype = "svg",
                                              savename = "sics_orig_tuned_cluster_mappings")

# Save the results locally
if save_file:
    if not os.path.exists(f"{datafolder}SICS_dec"):
        os.makedirs(f"{datafolder}SICS_dec")
    pd.DataFrame(y_pred_sics_orig_tuned).to_csv(f"{datafolder}y_pred_sics_orig_tuned.csv",
                                          header =False, index = False)
    pd.DataFrame(y_pred_mumc_orig_tuned).to_csv(f"{datafolder}y_pred_mumc_orig_tuned.csv",
                                          header = False, index = False)
    cluster_stab_sics_orig_tuned.to_csv(f"{datafolder}SICS_dec/"\
                                  "cluster_stab_sics_orig_tuned.csv")
    pd.DataFrame(Z_sics_orig_tuned).to_csv(f"{datafolder}SICS_dec/Z_sics_orig_tuned.csv")
    pd.DataFrame(y_pred_sics_orig_tuned).to_csv(f"{datafolder}SICS_dec/"\
                                          "y_pred_sics_orig_tuned.csv")
    pd.DataFrame(centroids_sics_orig_tuned).to_csv(f"{datafolder}SICS_dec/"\
                                             "centroids_sics_orig_tuned.csv")
    y_pred_stab_sics_orig_tuned.to_csv(f"{datafolder}SICS_dec/y_pred_stab_sics_orig_tuned.csv")
    model_sics_orig_tuned.save(f"{datafolder}SICS_dec/model_sics_orig_tuned")
    encoder_sics_orig_tuned.save(f"{datafolder}SICS_dec/encoder_sics_orig_tuned")