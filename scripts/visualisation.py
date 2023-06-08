# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:09:03 2022

@author: Jip de Kok

This file contains custom functions for the visualisation functionality.
"""
import sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import random
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import adjustText
from data_functions import scale_data
from matplotlib.patches import FancyBboxPatch


sns.set(rc={'figure.figsize':(11.7,8.27)},
        font_scale = 1.5)
sns.set_palette(sns.color_palette("deep"))
sns.set_style("whitegrid")


def cluster_heatmap(df, save=False, method = "default", outcome = "admission",
                    diagnosis_focus = True, annot = False, vmax = None,
                    SMR = False,
                    savename = "heatmap of outcomes per cluster",
                    filetype = "svg", title = None):
    '''
    Generates a heatmap illustrating the differences between various outcome
    variables in different clusters.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a predefined set of outcome variables and an additional
        column depicting the cluster a row belongs to. The outcome variables
        and their naming should be inline with the work of Castela Forte et. al
        2021
    save : boolean, optional
        boolean indicating whether to locally save the plot. The default is
        True.
    method : str, optional
        String defining what the color coding of the heatmap is based on. The 
        default is "default".
            default : colors correspond to percentage within cluster
            other : colors correspond to percentage per variable
    outcome: str, optional
        String specyfing which  set of outcome variables to load, possible 
        values are 'admission' and 'discharge'. The default is 'admission'
            admission: use the admission diagnosis (APACHE IV subgroup 1)
            discharge: use the discharge diagnosis
    diagnosis_focus: boolean, optional
        Boolean indicating whether to add an additonal plot that only shows 
        the diagnosis variables. This heatmap includes row and column total
        counts. If True, this is also the plot that will be saved locally
        if save=True.
    annot: boolean or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as data, then use this to annotate the heatmap instead of
        the data. Note that DataFrames will match on position, not index.
    vmax : float, optional
        value indicating the limit of the color bar. If None, it will be set to
        the highest value in the heatmap. The default is None.
    SMR : boolean, optional
        boolean indicating whether to include the standardised mortality rate.
        The default is False.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        DESCRIPTION. The default is "svg".

    Returns
    -------
    None.

    '''
    # Replace nan with "unknown"
    df.apacheIV_subgroup1[df.apacheIV_subgroup1.isna()] = "Unknown"
    
    # Store a complete copy of df
    df_total = df.copy()
    
    # Ensure that the cluster numbers start from 0 so they can be used as
    # indices as well
    it = 0
    while df_total['cluster'].min() != 0:
        df_total['cluster'] = df_total['cluster'] - 1
        it += 1
        if it > 100:
            sys.exit("Cluster labels should be integers starting from zero.")
    
    #================================DISCLAIMER!==============================#
    # The length of stay is computed on a linear scale, where the cluster with#
    # the highest mean los will always be 1, and the cluster with the lowest  #
    # los will alway be 0, this means that there will always appear to be a   #
    # the same magnitude of difference in los between clusters. This was done #
    # only to identically replicate the paper of Fore et. al 2021.            #
    #=========================================================================#
    # Filter out length of stay
    df = df_total[['icu_los', 'cluster']]
    # compute the mean los per cluster
    df = df.groupby(['cluster']).mean()
    # Linearly scale the mean los inbetween 0 and 1
    scaler = MinMaxScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), index= None, columns = None)
    # Store results in df_viz
    df_viz = df.copy()
    # Set variable name
    df_viz.columns = ["ICU length of stay (days)"]
    
    # Filter out outcome variables
    df = df_total[["mort_icu", "mort30", "mort90", "aki0", "aki1", "aki2",
                   "aki3", "vasoactive", "rrt", "cluster"]]
    # Combine aki2 and aki3 into one variables
    df.insert(5, "akie2or3",
              ((df["aki2"] == 1) | (df["aki3"] ==1)).astype(int))
    df.drop(["aki2", "aki3"], axis = 1, inplace = True)
    # Compute the total sum per cluster for ech variable
    df_sum = df.groupby(['cluster']).sum()
    # Compute the number of samples per cluster
    df_count = df.groupby("cluster").count()
    # Compute the ratio of each variable within each cluster
    df = df_sum / df_count
    # Set variable names
    renamer = {'mort_icu' : "in-ICU mortality", 'mort30' : "30-day mortality",
               'mort90' : "90-day mortality", 'aki0' : "No AKI during ICU stay",
               'aki1' : "AKI1", 'akie2or3' : "AKI 2 or 3",
               'vasoactive' : "Required vasoactive medication",
               'rrt' : "Required RRT"}
    df.rename(renamer, axis = 1, inplace = True)
    # Store results in df_viz
    df_viz = pd.concat([df_viz, df], axis = 1)
    
    if outcome == "discharge":
        # Primary discharge diagnoses.
        # Filter out primary discharge diagnoses variables
        df = df_total[["discharge diagnosis", "cluster"]]
        # Define 'other' category
        df.replace(["-", np.nan, 'Cancer', 'Hemato', 'Shock', 'Pain',
                    'Terminal', 'Heart'],
                   "other", inplace = True)
        df.replace(['Post transplant - heart', 'Post transplant - gastro',
                    'Post transplant - lung'],
                   "Post transplant - other", inplace = True)
        df.replace('Postoperative - trauma', 'Trauma', inplace = True)
        df.replace(['Electrolyte', 'Electrolytes'],
                   'Electrolyte disturbances', inplace = True)
        
        enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
        discharge_diag = pd.DataFrame(enc.fit_transform(
            df[["discharge diagnosis"]]))
        discharge_diag.columns = enc.get_feature_names_out()
        discharge_diag.columns = discharge_diag.columns.str.replace(
            "discharge diagnosis_", "")
        # Set variable names
        renamer = {'Hemodynamic' : "Hemodynamic instability/non-cardiac shock",
                   'Resp insufficiency' : "Respiratory failure"}
        discharge_diag.rename(renamer, axis = 1, inplace = True)
        
        
    elif outcome == "admission":
        # Primary discharge diagnoses.
        # Filter out primary discharge diagnoses variables
        df = df_total[["apacheIV_subgroup1", "cluster"]]

        enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
        discharge_diag = pd.DataFrame(enc.fit_transform(
            df[["apacheIV_subgroup1"]]))
        discharge_diag.columns = enc.get_feature_names_out()
        discharge_diag.columns = discharge_diag.columns.str.replace(
            "apacheIV_subgroup1_", "")
        discharge_diag.rename(renamer, axis = 1, inplace = True)
        
        # Compute outcome means
        outcomes_mean = df_total[["icu_los", "mort_icu", "vasoactive",
                                  "cluster"]].groupby("cluster").mean()
        
        # Add SMR if requested
        if SMR:
            outcomes_mean["apacheIV_mort"] = df_total[["apacheIV_mort",
                                      "cluster"]].groupby("cluster").mean()
            outcomes_mean["SMR"] = outcomes_mean["mort_icu"] / outcomes_mean["apacheIV_mort"]
    
    # Compute the ratio of each variable within each cluster
    discharge_diag["cluster"] = df.reset_index()['cluster']
    # Compute the total sum per cluster for ech variable
    df_sum = discharge_diag.groupby(['cluster']).sum()
    # Compute the number of samples per cluster
    if method == "default":
        df_count = discharge_diag.groupby("cluster").count()
    else:
        print("doing things a little different")
        df_count = discharge_diag.sum().drop("cluster")
    # Compute the ratio of each variable within each cluster
    discharge_diag = df_sum / df_count
    
    df_viz = pd.concat([df_viz, discharge_diag], axis = 1)
    
    if SMR:
        df_viz = pd.concat([df_viz, outcomes_mean[["apacheIV_mort", "SMR"]]], axis =1 )
    
    if outcome == "admission":
        if SMR:
            df_viz1 = df_viz.iloc[:,[0,1,7, 20, 21]]
        else:
            df_viz1 = df_viz.iloc[:,[0,1,7]]
    else:
        df_viz1 = df_viz.iloc[:,0:9]
        
        # Compute outcome means
        outcomes_mean = df_total[['icu_los', 'mort_icu', 'mort30', 'mort90',
                                  'aki0', 'aki1', 'aki2or3', 'vasoactive',
                                  'rrt', 'cluster']].groupby("cluster").mean()
        
    if SMR:
        df_viz2 = df_viz.iloc[:,9:20]
    else:
        df_viz2 = df_viz.iloc[:,9:]
        
    # Compute outcome totals of entire set
    if SMR:
        df_viz_outcome_total = pd.DataFrame(df_total[["icu_los", "mort_icu",
                                                     "vasoactive",
                                                     "apacheIV_mort"]].mean())
        df_viz_outcome_total.loc["SMR",:] = df_viz_outcome_total.loc[
            "mort_icu",:]/df_viz_outcome_total.loc["apacheIV_mort",:]
        df_viz_outcome_total_perc = df_viz_outcome_total.copy()
        df_viz_outcome_total_perc.iloc[0,0] = scaler.transform(np.array(
            df_viz_outcome_total_perc.iloc[0,0]).reshape(1,-1))
    else:
        df_viz_outcome_total = pd.DataFrame(df_total[["icu_los", "mort_icu",
                                                     "vasoactive"]].mean())
        df_viz_outcome_total_perc = df_viz_outcome_total.copy()
        df_viz_outcome_total_perc.iloc[0,0] = scaler.transform(np.array(df_viz_outcome_total_perc.iloc[0,0]).reshape(1,-1))
       
    
    # Plot the heatmaps
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (11,11),
                             gridspec_kw={'height_ratios': [1, 2.7],
                                          'width_ratios': [30, 1],
                                          'wspace': 0.1, 'hspace': 0.05},
                             constrained_layout=True)
    #fig.suptitle('Outcomes and clinical end-points per cluster',x=0.53, y=0.95)
    axes[0][0].set_title('Outcomes', fontsize=13, pad=10)
    axes[0][0].tick_params(axis='both', which='major', labelsize=12)
    axes[0][0].axes.get_xaxis().set_visible(False)
    axes[1][0].tick_params(axis='both', which='major', labelsize=12)
    if outcome == "admission":
        axes[1][0].set_title('Admission diagnosis', fontsize=13,
                             pad=10)
    else:
        axes[1][0].set_title('Primary discharge diagnosis category', fontsize=13,
                         pad=10)
    if annot:
        ax1 = sns.heatmap(df_viz1.transpose(),
                          cmap = sns.color_palette("Blues", as_cmap=True),
                          ax=axes[0][0], cbar_ax=axes[0][1],
                          annot = outcomes_mean.transpose(), fmt=".2f")
    else:
        ax1 = sns.heatmap(df_viz1.transpose(),
                          cmap = sns.color_palette("Blues", as_cmap=True),
                          ax=axes[0][0],
                          cbar_ax=axes[0][1], annot = annot, fmt=".2f")
    ax2 = sns.heatmap(df_viz2.transpose(),
                      cmap = sns.color_palette("Blues", as_cmap=True),
                      ax=axes[1][0], vmax = vmax,
                      cbar_ax=axes[1][1], annot = annot, fmt=".2f")
    axes[1][0].set_xticklabels(df_viz2.transpose().columns+1)
    
    # Added title if requested
    plt.suptitle(title)
    
    if diagnosis_focus:
        fig = plt.figure(figsize=(20,20))
        ax0 = plt.subplot2grid((20,20), (0,0), colspan=18, rowspan=4)
        ax1 = plt.subplot2grid((20,20), (5,0), colspan=18, rowspan=14)
        ax2 = plt.subplot2grid((20,20), (19,0), colspan=18, rowspan=1)
        ax3 = plt.subplot2grid((20,20), (5,18), colspan=2, rowspan=14)
        ax4 = plt.subplot2grid((20,20), (0,18), colspan=2, rowspan=4)
        
        mask = np.zeros_like(df_viz2)
        mask[np.tril_indices_from(mask)] = True
        
        cbar_ax = fig.add_axes([.91, .15, .03, .535])
        cbar_ax2 = fig.add_axes([.91, .73, .03, .15])
        
        sns.heatmap(df_viz1.transpose(), cmap="Blues", ax = ax0,
                         cbar_ax=cbar_ax2, annot = outcomes_mean.transpose(),
                         fmt=".2f")
       
        sns.heatmap(df_viz2.transpose(), ax=ax1, annot=df_sum.transpose(),
                   cmap="Blues", linecolor='b', cbar = True,
                   cbar_ax = cbar_ax, fmt = "g", vmax = vmax)
        ax0.xaxis.tick_top()
        ax0.set_xticklabels((df_viz2.transpose().columns + 1).astype(int))
        
        # Calculate vmax for the total bars per cluster and diagnosis
        if vmax == None:
            vmax_tot = df_sum.sum().sum() * df_viz2.max().max()
        else:
            vmax_tot = df_sum.sum().sum() * vmax
        
        # Make the total bars per cluster
        sns.heatmap((pd.DataFrame(df_sum.sum(axis = 1))).transpose(),
                   ax=ax2,  annot=annot, cmap='Blues', cbar=False,
                   xticklabels=False, yticklabels=False, fmt='g',
                   vmin = 0, vmax = vmax_tot)
        # Make total bars per diagnosis
        sns.heatmap(pd.DataFrame(df_sum.sum()), ax=ax3, vmin = 0,
                    vmax = vmax_tot, annot=annot, cmap='Blues',
                    cbar=False, xticklabels=False,
                    yticklabels=False, fmt='g')
       # Make total bars per outcome
        sns.heatmap(df_viz_outcome_total_perc, ax=ax4, annot=df_viz_outcome_total,
                   cmap='Blues', cbar=False, xticklabels=False, vmin = 0,
                   vmax = 1, yticklabels=False, fmt=".2f", cbar_ax = cbar_ax2)
        
        # Added title if requested
        plt.suptitle(title, y =0.93)
    
    if(save):
        plt.savefig(f'figures/{savename}.{filetype}',
                bbox_inches='tight')
    
    return


def cluster_heatmap_mumc(df, save=False, method = "default",
                         diagnosis_focus = True, annot = False, vmax = None,
                         SMR = False,
                         savename = "heatmap of outcomes per cluster",
                         filetype = "svg", title = None):
    '''
    Generates a heatmap illustrating the differences between various outcome
    variables in different clusters.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with a predefined set of outcome variables and an additional
        column depicting the cluster a row belongs to. The outcome variables
        and their naming should be inline with the work of Castela Forte et. al
        2021
    save : boolean, optional
        boolean indicating whether to locally save the plot. The default is
        True.
    method : str, optional
        String defining what the color coding of the heatmap is based on. The 
        default is "default".
            default : colors correspond to percentage within cluster
            other : colors correspond to percentage per variable
    diagnosis_focus: boolean, optional
        Boolean indicating whether to add an additonal plot that only shows 
        the diagnosis variables. This heatmap includes row and column total
        counts. If True, this is also the plot that will be saved locally
        if save=True.
    annot: boolean or rectangular dataset, optional
        If True, write the data value in each cell. If an array-like with the
        same shape as data, then use this to annotate the heatmap instead of
        the data. Note that DataFrames will match on position, not index.
    vmax : float, optional
        value indicating the limit of the color bar. If None, it will be set to
        the highest value in the heatmap. The default is None.
    SMR : boolean, optional
        boolean indicating whether to include the standardised mortality rate.
        The default is False.
    savename : str, optional
        String defining the naming of the save file if save=True. The default 
        is "heatmap of outcomes per cluster".
    filetype : str, optional
        DESCRIPTION. The default is "svg".

    Returns
    -------
    None.

    '''
    # Replace nan with "unknown"
    df.apacheIV_subgroup1[df.apacheIV_subgroup1.isna()] = "Unknown"
    
    # Store a complete copy of df
    df_total = df.copy()
   
    # Ensure that the cluster numbers start from 0 so they can be used as
    # indices as well
    it = 0
    while df_total['cluster'].min() != 0:
        df_total['cluster'] = df_total['cluster'] - 1
        it += 1
        if it > 100:
            sys.exit("Cluster labels should be integers starting from zero.")
    
    #================================DISCLAIMER!==============================#
    # The length of stay is computed on a linear scale, where the cluster with#
    # the highest mean los will always be 1, and the cluster with the lowest  #
    # los will alway be 0, this means that there will always appear to be a   #
    # the same magnitude of difference in los between clusters. This was done #
    # only to identically replicate the paper of Fore et. al 2021.            #
    #=========================================================================#
    # Filter out length of stay
    df = df_total[['icu_los', 'cluster']]
    # compute the mean los per cluster
    df = df.groupby(['cluster']).mean()
    # Linearly scale the mean los inbetween 0 and 1
    scaler = MinMaxScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), index= None, columns = None)
    # Store results in df_viz
    df_viz = df.copy()
    # Set variable name
    df_viz.columns = ["ICU length of stay (days)"]
    
    # Filter out outcome variables
    df = df_total[["mort_icu", "vasoactive", "cluster"]]
    # Compute the total sum per cluster for ech variable
    df_sum = df.groupby(['cluster']).sum()
    # Compute the number of samples per cluster
    df_count = df.groupby("cluster").count()
    # Compute the ratio of each variable within each cluster
    df = df_sum / df_count
    # Set variable names
    renamer = {'mort_icu' : "in-ICU mortality",
               'vasoactive' : "Required vasoactive medication"}
    df.rename(renamer, axis = 1, inplace = True)
    # Store results in df_viz
    df_viz = pd.concat([df_viz, df], axis = 1)
    
    if SMR:
        # Compute SMR variables
        df_SMR = df_total[["apacheIV_mort", "mort_icu", "cluster"]]
        df_SMR = df_SMR.groupby(['cluster']).mean()
        df_SMR["SMR"] = df_SMR["mort_icu"] / df_SMR["apacheIV_mort"]
        # Add SMR variables to df_viz
        df_viz = pd.concat([df_viz, df_SMR[["apacheIV_mort", "SMR"]]], axis = 1)
        df_viz.rename({'apacheIV_mort' : 'APACHE IV mortality'},
                      axis = 1, inplace = True)
    
    # Primary discharge diagnoses.
    # Filter out primary discharge diagnoses variables
    df = df_total[["apacheIV_subgroup1", "cluster"]]
    
    enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
    discharge_diag = pd.DataFrame(enc.fit_transform(
        df[["apacheIV_subgroup1"]]))
    discharge_diag.columns = enc.get_feature_names_out()
    discharge_diag.columns = discharge_diag.columns.str.replace(
        "apacheIV_subgroup1_", "")
    discharge_diag.rename(renamer, axis = 1, inplace = True)
    
    
    # Compute the ratio of each variable within each cluster
    discharge_diag["cluster"] = df['cluster']
    # Compute the total sum per cluster for ech variable
    df_sum = discharge_diag.groupby(['cluster']).sum()
    # Compute the number of samples per cluster
    if method == "default":
        df_count = discharge_diag.groupby("cluster").count()
    else:
        print("doing things a little different")
        df_count = discharge_diag.sum().drop("cluster")
    # Compute the ratio of each variable within each cluster
    discharge_diag = df_sum / df_count
    
    df_viz = pd.concat([df_viz, discharge_diag], axis = 1)
    
    # Separate outcomes and diagnoses
    if SMR:
        df_viz1 = df_viz.iloc[:,0:5]
        df_viz2 = df_viz.iloc[:,5:]
    else:
        df_viz1 = df_viz.iloc[:,0:3]
        df_viz2 = df_viz.iloc[:,3:]
        
    # Compute outcome means
    if SMR:
        outcomes_mean = df_total[["icu_los", "mort_icu", "vasoactive",
                                  "apacheIV_mort",
                                  "cluster"]].groupby("cluster").mean()
        outcomes_mean = pd.concat([outcomes_mean, df_viz1["SMR"]], axis = 1)
    else:
        outcomes_mean = df_total[["icu_los", "mort_icu", "vasoactive",
                              "cluster"]].groupby("cluster").mean()
        
    # Compute the totals of the entire set
    if SMR:
        df_viz_outcome_total = pd.DataFrame(df_total[["icu_los", "mort_icu",
                                                      "vasoactive",
                                                      "apacheIV_mort"]].mean())
        df_viz_outcome_total.loc["SMR",:] = df_viz_outcome_total.loc[
            "mort_icu",:]/df_viz_outcome_total.loc["apacheIV_mort",:]
        df_viz_outcome_total_perc = df_viz_outcome_total.copy()
        df_viz_outcome_total_perc.iloc[0,0] = scaler.transform(np.array(df_viz_outcome_total_perc.iloc[0,0]).reshape(1,-1))
    else:
        df_viz_outcome_total = pd.DataFrame(df_total[["icu_los", "mort_icu",
                                                      "vasoactive"]].mean())
        df_viz_outcome_total_perc = df_viz_outcome_total.copy()
        df_viz_outcome_total_perc.iloc[0,0] = scaler.transform(np.array(df_viz_outcome_total_perc.iloc[0,0]).reshape(1,-1))
        
    
    # Plot the heatmaps
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (11,11),
                             gridspec_kw={'height_ratios': [1, 2.7],
                                          'width_ratios': [30, 1],
                                          'wspace': 0.1, 'hspace': 0.05},
                             constrained_layout=True)
    #fig.suptitle('Outcomes and clinical end-points per cluster',x=0.53, y=0.92)
    axes[0][0].set_title('Outcomes', fontsize=13, pad=10)
    axes[0][0].tick_params(axis='both', which='major', labelsize=12)
    axes[0][0].axes.get_xaxis().set_visible(False)
    axes[1][0].tick_params(axis='both', which='major', labelsize=12)
    axes[1][0].set_title('Admission diagnosis', fontsize=13,
                         pad=10)
    if annot:
        ax1 = sns.heatmap(df_viz1.transpose(), cmap="Blues", ax=axes[0][0],
                          cbar_ax=axes[0][1], annot = outcomes_mean.transpose(),
                          fmt=".2f")
    else:
        ax1 = sns.heatmap(df_viz1.transpose(), cmap="Blues", ax=axes[0][0],
                              cbar_ax=axes[0][1], annot = False, fmt=".2f")
    ax2 = sns.heatmap(df_viz2.transpose(), cmap = "Blues", ax=axes[1][0],
                      cbar_ax=axes[1][1], annot = annot, fmt=".2f")
    axes[1][0].set_xticklabels((df_viz2.transpose().columns+1).astype(int))
    
    plt.suptitle(title)
    
    if(save):
        plt.savefig(f'figures/{savename}.svg',
                bbox_inches='tight')
    
    if diagnosis_focus:
        fig = plt.figure(figsize=(20,20))
        ax0 = plt.subplot2grid((20,20), (0,0), colspan=18, rowspan=4)
        ax1 = plt.subplot2grid((20,20), (5,0), colspan=18, rowspan=14)
        ax2 = plt.subplot2grid((20,20), (19,0), colspan=18, rowspan=1)
        ax3 = plt.subplot2grid((20,20), (5,18), colspan=2, rowspan=14)
        ax4 = plt.subplot2grid((20,20), (0,18), colspan=2, rowspan=4)
        
        mask = np.zeros_like(df_viz2)
        mask[np.tril_indices_from(mask)] = True
        
        cbar_ax = fig.add_axes([.91, .15, .03, .535])
        cbar_ax2 = fig.add_axes([.91, .73, .03, .15])
        
        sns.heatmap(df_viz1.transpose(), cmap="Blues", ax = ax0,
                          cbar_ax=cbar_ax2, annot = outcomes_mean.transpose(),
                          fmt=".2f", vmax=1)
        
        sns.heatmap(df_viz2.transpose(), ax=ax1, annot=df_sum.transpose(),
                    cmap="Blues", linecolor='b', cbar = True,
                    cbar_ax = cbar_ax, fmt = "g", vmax = vmax)
        ax0.xaxis.tick_top()
        ax0.set_xticklabels((df_viz2.transpose().columns + 1).astype(int))
        
        # Calculate vmax for the total bars per cluster and diagnosis
        if vmax == None:
            vmax_tot = df_sum.sum().sum() * df_viz2.max().max()
        else:
            vmax_tot = df_sum.sum().sum() * vmax
        
        # Make the total bars per cluster
        sns.heatmap((pd.DataFrame(df_sum.sum(axis = 1))).transpose(),
                   ax=ax2,  annot=annot, cmap='Blues', cbar=False,
                   xticklabels=False, yticklabels=False, fmt='g',
                   vmin = 0, vmax = vmax_tot)
        # Make total bars per diagnosis
        sns.heatmap(pd.DataFrame(df_sum.sum()), ax=ax3, vmin = 0,
                    vmax = vmax_tot, annot=annot, cmap='Blues',
                    cbar=False, xticklabels=False,
                    yticklabels=False, fmt='g')
       # Make total bars per outcome
        sns.heatmap(df_viz_outcome_total_perc, ax=ax4, annot=df_viz_outcome_total,
                   cmap='Blues', cbar=False, xticklabels=False, vmin = 0,
                   vmax = 1, yticklabels=False, fmt=".2f", cbar_ax = cbar_ax2)
        
        plt.suptitle(title, y =0.93)
        
        if(save):
            plt.savefig(f'figures/{savename}_diagnosis_focus.{filetype}',
                    bbox_inches='tight')
    
    return


def compare_variables(df1, df2, plot_type = "boxplot",
                      data_names = ["sics", "MUMC+"], 
                      seperator = ["get_mean", "get_var", "other"],
                      sep_type = "substrings", save = False,
                      filetype = "svg"):
    """
    compare_variables produces boxplots of all variables in the data.

    Parameters
    ----------
    df1 : pandas.DataFrame
        DataFrame of the first data set where rows are samples and columns are
        variables.
    df2 : pandas.DataFrame
        DataFrame of the second data set where rows are samples and columns are
        variables. The columns should be identical to df1, rows can differ.
    plot_type : str, optional
        String specifying the type of plot to construct. Possible values are
        "boxplot" and "violinplot". The default is "boxplot".
    data_names : ndarray, optional
        An array of strings where each string corresponds to the name of the
        data source of df1 and df2 respectively. The default is 
        ["sics", "MUMC+"].
    seperator : ndarray, optional
        An array of strings where each string corresponds to a characyer 
        sequence based upon which the variables are split into different plots.
        This can particularly useful if there are too many variables to
        visualise in one plot. The string "other" can also be added, which
        means that variables which do not contain any of the defined seperators
        will be combined in one additional plot. If "other" is included in 
        seperator, it MUST be the last element in seperator. The default is 
        ["get_mean", "get_var", "other"]. If sep_type='max_value', you can
        supply an array floats, where each value corresponds to a cutoff of the
        maximum value of a variable to be allowed in a plot.
    sep_type : str, optional,
        String specifying the method by which to split to variables into
        seperate plots. Possible values are 'substrings' and 'max_value'
            subtrings: Use substrings specified in seperator\n
            max_value: use max values specified in seperator. Values which
            are higher than the highest max value supplied in seperator
            will not be plotted, so make sure the highest max value in seperator
            is at least as high as the highest value in your data. When unsure,
            set to infinity.
    save : boolean, optional
        boolean indicating whether to locally save the plot. The default is
        True.
    filetype : str, optional
        DESCRIPTION. The default is "svg".

    Returns
    -------
    None.

    """
    df_plot1 = pd.melt(df1)
    df_plot1["data"] = data_names[0]
    df_plot2 = pd.melt(df2)
    df_plot2["data"] = data_names[1]
    
    # Seperate binary variables
    bin_vars = df1.columns[df1.nunique() == 2]
    df_plot1_bin = df_plot1[df_plot1.variable.isin(bin_vars)]
    df_plot2_bin = df_plot2[df_plot2.variable.isin(bin_vars)]
    # Combine binary variables of both data sets into one
    df_plot = df_plot1_bin.append(df_plot2_bin)
    # Compute how often variable for each dataset is either 1 or 0
    df_plot = df_plot.groupby(["variable", "value", "data"]).size().reset_index()
    df_plot.rename(columns = {0 : "ratio"}, inplace = True)
    # For dataset 1, compute how much of a variable is either 1 or 0 (percentage)
    df_plot.loc[df_plot.data == data_names[0], "ratio"] = df_plot[
        df_plot.data == data_names[0]].ratio/len(df1)
    # For dataset 2, compute how much of a variable is either 1 or 0 (percentage)
    df_plot.loc[df_plot.data == data_names[1], "ratio"] = df_plot[
        df_plot.data == data_names[1]].ratio/len(df2)
    # sort dataframe on dataset name
    df_plot = df_plot.sort_values(["data", "variable"],
                                  ascending=False).reset_index(drop = True)
    
    # Split 0 and 1 values (bins) into distinct dataframes
    df_plot_bin0 = df_plot[df_plot.value == 0].reset_index(drop=True)
    df_plot_bin1 = df_plot[df_plot.value == 1].reset_index(drop=True)
    
    # Add ratio of bin0 onto bin1 such that it will end up on top of bin0
    # in the stacked bar chart
    df_plot_bin1.ratio = df_plot_bin1.ratio + df_plot_bin0.ratio
    
    # Plot a stacked bar chart of the ratio of binary variables
    plt.figure()
    bar1 = sns.barplot(y="variable",  x="ratio", data=df_plot_bin1,
                       hue = "data",
                       palette = sns.color_palette("deep")[0:2])
    bar1.set_yticklabels(bar1.get_yticks(), ha="right",
                          va = "top")
    bar2 = sns.barplot(y="variable",  x="ratio", data=df_plot_bin0,
                       hue = "data",
                       palette = list([sns.color_palette("deep")[9],
                                       sns.color_palette("deep")[3]]))
    bar2.set_yticklabels(['worsened respiratory condition after 24h',
                          'history of respiratory insufficiency',
                          'history of myocardial infarction',
                          'history of metastatic disease',
                          'history of hematological malignancy',
                          'history of immune insufficiency',
                          'history of diabetes',
                          'previous dialysis',
                          'history of chronic obstructive pulmonary disease',
                          'history of cirrhosis',
                          'history of chronic kidney disease',
                          'history of cardiovascular disease',
                          'previous admission to ICU',
                          'mechanical ventilation at admission',
                          'mechanical ventilation after first 24h',
                          'gender',
                          'central venous pressure',
                          'history of atrial fibrillation',
                          'surgical admission'])


    plt.xlim([0,1])
    plt.xlabel("yes/no ratio")
    plt.ylabel("")
    
    
    ld1 = mpatches.Patch(color=sns.color_palette("deep")[0])
    ld2 = mpatches.Patch(color=sns.color_palette("deep")[9])
    ld3 = mpatches.Patch(color=sns.color_palette("deep")[1])
    ld4 = mpatches.Patch(color=sns.color_palette("deep")[3])
    bottom_bar = mpatches.Patch(color='lightblue', label=1)
    plt.legend(labels=["SICS - yes",
                       "SICS - no",
                       "MUMC+ - yes",
                       "MUMC+ - no"],
               handles=[ld1,ld2,ld3,ld4],
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"figures/bar plot comparison between"\
                    f" {data_names[0]} and {data_names[1]}"\
                    f"binary variables.{filetype}",
                    bbox_inches='tight')
    
    # Remove binary variables from dataset
    df_plot1 = df_plot1[~df_plot1.variable.isin(bin_vars)]
    df_plot2 = df_plot2[~df_plot2.variable.isin(bin_vars)]
    
    
    
    # Store max value per variable
    if sep_type == "max_values":
        seperator.sort()
        df_plot_combined = df_plot1.append(df_plot2)
        var_max = df_plot_combined.groupby("variable").max("value")
    
    for sep in seperator:
        if sep_type == "substrings":
            if sep == "other":
                df_plot1_sep = df_plot1[~df_plot1.variable.str.contains(
                    '|'.join(seperator[:-1]))]
                df_plot2_sep = df_plot2[~df_plot2.variable.str.contains(
                    '|'.join(seperator[:-1]))]
            else:
                df_plot1_sep = df_plot1[df_plot1.variable.str.contains(sep)]
                df_plot2_sep = df_plot2[df_plot2.variable.str.contains(sep)]
        elif sep_type == "max_values":
            var_list = var_max[var_max.value <= sep]
            df_plot1_sep = df_plot1[df_plot1.variable.isin(var_list.index)]
            df_plot2_sep = df_plot2[df_plot2.variable.isin(var_list.index)]
            var_max[var_max.value <= sep] = np.Infinity
        else:
            sys.exit("Unsupported sep_type. Should be 'substrings' or"\
                     " 'max_values'.")
        
        df_plot = df_plot1_sep.append(df_plot2_sep)
        
        plt.figure()
        if plot_type == "boxplot":
            ax = sns.boxplot(data = df_plot, x = "value", y = "variable",
                             hue = "data")
        elif plot_type == "violinplot":
            ax = sns.violinplot(data = df_plot, x = "value", y = "variable",
                                hue = "data")
        else :
            sys.exit("Unsupported plot_type. Should be 'boxplot' or"\
                     " 'violinplot'.")
        ax.xaxis.grid(True)
        ax.yaxis.grid(True)
        ax.set(xlabel = None, ylabel = None)
        plt.tight_layout()
        
        for i in ax.collections:
            if type(i) == matplotlib.collections.PolyCollection:
                # Get color of current violin
                c = i.get_facecolor()
                # Set edge color to match the face color
                i.set_edgecolor(c)
                # Set alpha to 0.6
                c[0][3] = 0.6
                # Apply the transparent color to the face color
                i.set_facecolor(c)

        if save:
            plt.savefig(f"figures/{plot_type} comparison between"\
                        f" {data_names[0]} and {data_names[1]}"\
                        f"{sep_type}_{sep}.{filetype}",
                        bbox_inches='tight')
    
    
        
    return


def cluster_boxplots(df, var = "icu_los", plot_type = "boxplot", save = False,
                     savename = "cluster_variable_comparison",
                     filetype = "svg"):
    '''
    Create boxplots per cluster, comparing the distribution of one variable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containg samples (rows), the variable of interest, and a
        variable containing the cluster label for each sample.
    var : str, optional
        The variable of interest that you want to compare between clusters.
        The default is "icu_los".
    plot_type : str, optional
        Type of plot to construct. Can be "boxplot" or "violinplot". The
        default is "boxplot".
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
    None.

    '''
    # Store a complete copy of df
    df_total = df.copy()
   
    # Ensure that the cluster numbers start from 0 so they can be used as
    # indices as well
    it = 0
    while df_total['cluster'].min() != 0:
        df_total['cluster'] = df_total['cluster'] - 1
        it += 1
        if it > 100:
            sys.exit("Cluster labels should be integers starting from zero.")
            
    # Filter out variable of interest
    df_sub = df_total[[var, 'cluster']]
    
    plt.figure()
    if plot_type == "boxplot":
        sns.boxplot(df_sub.iloc[:,1], df_sub.iloc[:,0])
    if plot_type == "violinplot":
        sns.violinplot(df_sub.iloc[:,1], df_sub.iloc[:,0])
    plt.tight_layout()    
    
    if(save):
        plt.savefig(f"figures/{savename}.{filetype}")
        
    return
        

def pca_score_plot(df, df_project = None, n_components = None, seed = None,
                   scale = False, plot_3D = False, biplot = False,
                   multi_plot = False,
                   dataset_names = ["dataset1", "dataset2"],
                   threshold = 0.2, components = [1, 2, 3],
                   save = False, savename = "PCA", filetype = "svg"):
    '''
    Create a PCA score plot

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data you want to perform PCA on.
    df_project : pandas.DataFrame, optional
        DataFrame you want to project onto the PCA of df. If you only want to
        only plot the PCA plot of df without additional projection than this
        should be None. The default is None.
    n_components : int, float or 'mle', optional
        Number of components to keep. if n_components is not set all components
        are kept The default is None.
    seed : int, optional
        Random seed for reproducibility. The default is None.
    scale : boolean, optional
        Boolean indicating whether data should be normalised prior to PCA. The
        default is False.
    plot_3D : boolean, optional
        Boolean indicating whether a 3D PCA score plot should be made. The
        default is False.
    biplot : boolean, optional
        Boolean indicating whether a biplot should be made. The default is False.
    multi_plot : boolean, optional
        Boolean indicating whether a multiplot should be made to visualise
        multiple principle components. The default is False.
    dataset_names : ndarray, optional
        Array of strings, specifying the names of the two datasets. The default
        is ["dataset1", "dataset2"].
    threshold : float, optional
        The minimum loading for a variable to be included in the biplot. The
        default is 0.2.
    components : ndarray, optional
        Array of integers specifying which components to plot. Should contain
        three values if plot_3D=True. The default is [1, 2, 3].
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
    length : TYPE
        DESCRIPTION.

    '''
    
    if scale == True:
        df, _ = scale_data(df, method = "standardise")
        if not df_project is None:
            df_project, _ = scale_data(df_project, method = "standardise")
    
    
    # Initialise PCA
    pca = PCA(n_components = n_components, random_state = seed)
    # fit the PCA model
    pc_fit = pca.fit(df)
    pc = pc_fit.transform(df)
    pc_df = pd.DataFrame(data = pc, columns = ["PC" + str(i) for i in 
                                               range(1, pc.shape[1] +1)])
    
    # Add column depicting dataset name
    pc_df['dataset'] = dataset_names[0]
    pc_df['dataset_color'] = sns.color_palette().as_hex()[0]
    
    if df_project is not None:
        pc2 = pc_fit.transform(df_project)
        pc_df2 = pd.DataFrame(data = pc2, columns = ["PC" + str(i) for i in
                                                     range(1, pc.shape[1] +1)])
        pc_df2['dataset'] = dataset_names[1]
        pc_df2['dataset_color'] = sns.color_palette().as_hex()[1]
        pc_df = pd.concat([pc_df2, pc_df])
        
    # Generate 2D PCA score plot
    plt.figure()
    scoreplot = sns.scatterplot(pc_df[f"PC{components[0]}"],
                                pc_df[f"PC{components[1]}"],
                                alpha = 0.6, s = 100, hue = pc_df["dataset"],
                                hue_order = [np.unique(pc_df["dataset"])[1],
                                             np.unique(pc_df["dataset"])[0]])
    
    # Set figure size based on length of the scores
    pc1_length = np.diff(scoreplot.get_xlim())[0]
    pc2_length = np.diff(scoreplot.get_ylim())[0]
    scoreplot.figure.set_figwidth(11.7)
    scoreplot.figure.set_figheight((pc2_length/pc1_length)*11.7)
    scoreplot.set_xlabel(
        f"PC{components[0]} "\
        f"({pca.explained_variance_ratio_[components[0]-1].round(3)*100}%)")
    scoreplot.set_ylabel(
        f"PC{components[1]} "\
        f"({pca.explained_variance_ratio_[components[1]-1].round(3)*100}%)")
    
    
    if df_project is None:
        scoreplot.legend_.remove()
    plt.tight_layout()
    
    if save == True:
        plt.savefig(f"figures/{savename}_2D.{filetype}")
    
    if(plot_3D):
        fig_3D = plt.figure(figsize=(7,5))
        ax = fig_3D.add_subplot(111, projection='3d')
        ax.scatter(pc_df[f"PC{components[0]}"], pc_df[f"PC{components[1]}"],
                   pc_df[f"PC{components[2]}"],
                   c = pc_df["dataset_color"],
                   label = pc_df["dataset"], alpha = 0.6, s = 100)
        ax.set_xlabel(
            f"PC{components[0]} "\
            f"({pca.explained_variance_ratio_[components[0]-1].round(3)*100}%)")
        ax.set_ylabel(
            f"PC{components[1]} "\
            f"({pca.explained_variance_ratio_[components[1]-1].round(3)*100}%)")
        ax.set_zlabel(
            f"PC{components[2]} "\
            f"({pca.explained_variance_ratio_[components[2]-1].round(3)*100}%)")
    
    # Generate biplot if requested
    if biplot == True:
        # Create function that computes the length of a loading vector 
        # -- used by biplot() --
        def compute_loading_length(x):
            length = np.sqrt(np.square(x[0]) + np.square(x[1]))
            return length
        
        # Create function to generate biplot
        def biplot(scores, loadings,labels=None, threshold = threshold):  
            # Determine min and max values for the x and y scale
            all_x_values = pd.concat([scores.iloc[:,0], loadings.iloc[:,0]])
            all_y_values = pd.concat([scores.iloc[:,1], loadings.iloc[:,1]])
            x_min = np.min(all_x_values) - 0.1
            x_max = np.max(all_x_values) + 0.1
            y_min = np.min(all_y_values) - 0.1
            y_max = np.max(all_y_values) + 0.1
            
            # Compute the length of each loading vector
            loading_length = loadings.apply(compute_loading_length, axis =1)
            # Only keep loadings bigger than the threshold and their according
            # variable labels
            labels = labels[loading_length >= threshold]
            loadings = loadings[loading_length >= threshold]
            n = loadings.shape[0]
            
            # Initiate plot
            fig = sns.scatterplot(scores.iloc[:,0], scores.iloc[:,1], s = 100,
                                  alpha = 0.8)
            # Iniate text (will contain labels for the arrows)
            texts = []
            # Draw the loading vectors as arrows, and add a text label to it
            for i in range(n):
                arrow = mpatches.FancyArrowPatch((0, 0),(loadings.iloc[i,0],
                                                         loadings.iloc[i,1]),
                                                 **dict(arrowstyle="Fancy,tail_width=2,head_width=8,head_length=7"),
                                                 color = "black", alpha = 0.4,
                                                 shrinkB=10)
                fig.axes.add_patch(arrow)
                texts.append(plt.text(loadings.iloc[i,0], loadings.iloc[i,1], labels[i],
                          color = 'red', fontsize = 12, weight='bold'))
            
            plt.xlabel(f"PC{components[0]} "\
            f"({pca.explained_variance_ratio_[components[0]-1].round(3)*100}%)")
            plt.ylabel(f"PC{components[1]} "\
            f"({pca.explained_variance_ratio_[components[1]-1].round(3)*100}%)")
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            # Enforce x and y on same scale
            plt.gca().set_aspect("equal")
            plt.tight_layout()
            # Make sure text labels don't overlap
            adjustText.adjust_text(texts,
                                   autoalign = "y",
                                   only_move = {"text" : "y"},
                                   avoid_points = False,
                                   avoid_self = False,
                                   force_text = (0,8),
                                   force_points = (0,0),
                                   force_objects = (0,0))
            
        
        # Define PC scores of interest
        scores = pd.DataFrame(pc[:,np.subtract(components, 1)])
        # Define PC loadings
        loadings = pd.DataFrame(pca.components_[
            np.subtract(components, 1), :]).transpose()
        
        # Plot the biplot
        plt.figure()
        biplot(scores, loadings, labels = pca.feature_names_in_)
        if save == True:
            plt.savefig(f"figures/{savename}_biplot.{filetype}")
            
            
    if multi_plot == True:
        multiplot1 = sns.pairplot(pc_df.iloc[:,[0,1,2,3,4,5,6,7,-2]],
                                  hue = "dataset", diag_kind=None,
                                  corner = True, height = 1, aspect = 2,
                                  plot_kws=dict(alpha=0.4, s=5),
                                  hue_order = [np.unique(pc_df["dataset"])[1],
                                               np.unique(pc_df["dataset"])[0]])
        multiplot1.figure.set_figwidth(23.4)
        multiplot1.figure.set_figheight(23.4)
        if save == True:
            multiplot1.savefig(f"figures/{savename}_multiplot1.{filetype}",
                               dpi = 900)
        
        
        multiplot2 = sns.pairplot(pc_df[pc_df.dataset ==
                                        dataset_names[0]].iloc[:,0:8],
                                  corner=True, diag_kind=None, height = 1,
                                  aspect = 1, plot_kws=dict(alpha=0.4, s=5))
        multiplot2.map_lower(sns.kdeplot, levels=4, color=".2")
        multiplot2.figure.set_figwidth(23.4)
        multiplot2.figure.set_figheight(24.4)
        if save == True:
            multiplot2.savefig(f"figures/{savename}_multiplot2.{filetype}",
                               dpi = 900)
    
    return

    
def cluster_pca(Z, labels, stability = None, n_components = None, title = None,
                plot_3D = False, scale = False, multi_plot = False, save=False,
                savename = "Cluster PCA", filetype = "svg"):
    '''
    plot PCA score plots of with clustering labels depicted by color.

    Parameters
    ----------
    Z : pandas.DataFrame
        DataFrame containing the latent features (columns) for all samples (rows).
    labels : ndarray or pandas.DataFrame with one column
        Array of cluster memberships per sample.
    stability : boolean, optional
        Boolean indicating whether to include sample stability information. The
        default is None.
    n_components : int, float or 'mle', optional
        Number of components to keep. if n_components is not set all components
        are kept The default is None.
    plot_3D : boolean, optional
        Boolean indicating whether a 3D PCA score plot should be made. The
        default is False.
    scale : boolean, optional
        Boolean indicating whether data should be normalised prior to PCA. The
        default is False.
    multi_plot : boolean, optional
        Boolean indicating whether a multiplot should be made to visualise
        multiple principle components. The default is False.
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
    None.

    '''
    if scale:
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)
    
    # plot PCA scores
    # Initiate PCA model
    pca = PCA(n_components = n_components)

    # fit the PCA model
    pca_fit = pca.fit_transform(Z)

    pc_df = pd.DataFrame(data = pca_fit, columns = ["PC" + str(i) for i in
                                                    range(1,pca_fit.shape[1] +1)])
    pc_df["Cluster"] = labels
    
    color_map = ListedColormap(sns.color_palette(n_colors=len(np.unique(
        labels))).as_hex())
    
    if(plot_3D):
        fig_3D = plt.figure(figsize=(7,5))
        ax = fig_3D.add_subplot(111, projection='3d')
        ax.set_title('PCA score plot of Z with clusters', fontsize=13, pad=10)
        if stability is None:
            ax.scatter(pc_df["PC1"], pc_df["PC2"], pc_df["PC3"], c = pc_df["Cluster"],
                       alpha = 0.6, s = 100, label = pc_df["Cluster"],
                       cmap = color_map)
        else:
            ax.scatter(pc_df["PC1"], pc_df["PC2"], pc_df["PC3"], c = stability,
                       alpha = 0.6, s = 100,
                       cmap = sns.blend_palette(
                           [sns.color_palette("deep")[0],
                            sns.color_palette("deep")[1]], as_cmap = True))

    
    plt.figure()
    if stability is None:
        sns.scatterplot(pc_df["PC1"], pc_df["PC2"], hue = pc_df["Cluster"],
                        alpha = 0.6, s = 100, palette = sns.color_palette(
                            n_colors=len(np.unique(labels))))
    else:
        ax = sns.scatterplot(pc_df["PC1"], pc_df["PC2"],
                             hue = stability, alpha = 0.8, s = 100,
                             style = pc_df["Cluster"],
                             palette = sns.blend_palette(
                                 [sns.color_palette("deep")[0],
                                  sns.color_palette("deep")[1]],
                                 as_cmap = True))
        h,l = ax.get_legend_handles_labels()
        ax.legend(h[int(len(h)/2):],l[int(len(l)/2):], loc='upper left',
                  title = "Cluster")
        norm = plt.Normalize(stability.min(),
                             stability.max())
        sm = plt.cm.ScalarMappable(cmap= sns.blend_palette(
            [sns.color_palette("deep")[0],
             sns.color_palette("deep")[1]],
            as_cmap = True), norm=norm)
        ax.figure.colorbar(sm)
    plt.suptitle(title)
    if(save):
        plt.savefig(f"figures/{savename}.{filetype}")
    
    if multi_plot == True:
        multiplot1 = sns.pairplot(pc_df.iloc[:,[0,1,2,3,4,-1]],
                                  diag_kind=None,
                                  hue = "Cluster",
                                  palette = sns.color_palette("deep")[0:6],
                                  corner = True, height = 2, aspect = 1,
                                  plot_kws=dict(alpha=0.5, s=20))
        plt.suptitle(title)
        plt.tight_layout()
        
        if(save):
            plt.savefig(f"figures/{savename}_multiplot.{filetype}")

    
    


def plot_patient_timecourse(subset = 10):
    '''
    Generate a line plot of the distribution of measurements over time for 
    a set number of random samples.

    Parameters
    ----------
    subset : integer, optional
        number specifying how many random samples to plot. The default is 10.

    Returns
    -------
    None.

    '''
    # Load data sets
    df = pd.read_csv("data/final_merged_MUMC.csv")
    
    # Convert time variables to datetime format
    df["chartTime"] = pd.to_datetime(df["chartTime"])
    
    # Set function to compute time relative to first measurement
    def compute_rel_time(x):
        x = x - x.min()
        return(x)
    
    # Get relative time to first measurement per sample
    df_timings = pd.DataFrame(df.groupby("encounterId")["chartTime"].apply(
        compute_rel_time))
    df_timings["encounterId"] = df["encounterId"]
    
    # Subset the samples
    random_samples = random.choices(df_timings["encounterId"], k=subset)
    df_timings = df_timings[df_timings["encounterId"].isin(random_samples)]
    
    # Turn time unit into hours
    df_timings["chartTime"] = df_timings["chartTime"].astype('timedelta64[h]')
    
    # Turn encounterID into a string so it's not handled as a numeric value
    df_timings["encounterId"] = df_timings["encounterId"].astype(str)
    plt.figure()
    ax = sns.kdeplot(data = df_timings, x = "chartTime", hue = "encounterId",
                     alpha = 0.6)
    ax.set(xlim = [0, df_timings["chartTime"].max()])


    
def plot_missingness_pattern(df):
    '''
    Plot the missingness pattern in the data. The pattern describes how often
    two variables are missing together in conjunction. 1 indicates that if
    the variable on the x-axis is missing, the variable on the y-axis is also
    always missing. 0 indicates that variables are never both missing in the
    same sample.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of which you want to visualise the missingnes per variable.

    Returns
    -------
    None.

    '''

    missing_vars = df.columns[df.isna().any()]
    
    def missing_overlap(var1, var2):
        
        # Compute how often var2 is missing when var1 is missing
        var1_missing = np.sum(var1.isna())
        var2_missing = var2[var1.isna()].isna().sum()
        if(var1_missing == 0):
            overlap = np.nan
        else:
            overlap = np.round(var2_missing/var1_missing,2)
        
        return(overlap)
    
    output = pd.DataFrame(index = missing_vars, columns = missing_vars,
                          dtype = float)
    for x in missing_vars:
        for y in missing_vars:
            output.loc[x, y] = missing_overlap(df[x], df[y])
    
    
    plt.figure(figsize = (18,10))
    
    sns.set(font_scale = .7)
    ax1 = sns.heatmap(output, cmap='Blues', linewidths=.5, linecolor="grey",
                      annot = True, annot_kws={"fontsize":8})
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45, ha="right")
    ax1.set
    
    plt.tight_layout()
    
    plt.savefig(f'figures/missingess_pattern.svg', bbox_inches='tight')
    plt.savefig(f'figures/missingess_pattern.png', bbox_inches='tight')


def plot_stability(df_stab, save = False, filetype = "svg", mutation = 1,
                   stability_type = "sample", title = None,
                   savename = "stability histogram"):
    '''
    Generate plots of the clustering stability.

    Parameters
    ----------
    df_stab : pandas.DataFrame
        DataFrame containing the computed stability scores per cluster or per
        sample.
    save : Boolean, optional
        Boolean indicating whether to save the plot. The default is False.
    filetype : String, optional
        String specifying the filetype to save the plot with. The default is 
        "svg".
    mutation : float, optional
        A number by which the mutations aspect for the rounding of the bars
        gets multiplied, if the rounding of bars in the visualisation are off,
        try changing this parameter. The default is 1.
    stability_type : str, optional
        String specifying the stability, either cluster- or sample-wise. If
        the the clusters were mapped based on centroids or overlap, then the
        results are sample-wise, if jaccard was used then the stabilities are
        cluster-wise. Possible values are "sample" and "cluster". The default 
        is "sample".
    savename : str, optional
        String specifying the name of the file the plot will be saved to if
        save = True. The default is "stability histogram".

    Returns
    -------
    Pyplot object.

    '''
    
    
    if stability_type == "sample":
    
        plt.figure(figsize = (10,8))
        ax1 = sns.histplot(df_stab*100, kde = False, legend = False,binrange = [0,100],
                           binwidth = 5, element = "bars",
                           color = sns.color_palette("deep")[0],
                           alpha = 0.25,
                           edgecolor = sns.color_palette("deep")[0],
                           linewidth = 2)
        
        
        ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_xlim([0,101])
        ax1.set_xlabel("Stability (%)")
        ax1.set_ylabel("Number of samples")
        ax1.grid(axis = 'y')
        plt.suptitle(title)
        
        for patch in reversed(ax1.patches):
            bb = patch.get_bbox()
            color = patch.get_facecolor()
            ec = patch.get_edgecolor()
            width = bb.get_points()[[1]][0][0] - bb.get_points()[[0]][0][0]
            height = bb.get_points()[[1]][0][1] - bb.get_points()[[0]][0][1]
            if height >= 15:
                p_bbox = FancyBboxPatch((bb.get_points()[[0]][0][0],
                                         bb.get_points()[[0]][0][1]),
                                        width, height,
                                        boxstyle="Round,pad=0,rounding_size=1",
                                        ec=ec, fc=color, lw = 2,
                                        mutation_aspect=5 * mutation,
                                        mutation_scale = 2)
                patch.remove()
                ax1.add_patch(p_bbox)
            elif height >=2:
                p_bbox = FancyBboxPatch((bb.get_points()[[0]][0][0],
                                         bb.get_points()[[0]][0][1]),
                                        bb.get_points()[[1]][0][0] - bb.get_points()[[0]][0][0],
                                        bb.get_points()[[1]][0][1] - bb.get_points()[[0]][0][1],
                                        boxstyle="Round,pad=0,rounding_size=1",
                                        ec=ec, fc=color, lw = 2,
                                        mutation_aspect=5/(15-height) * mutation,
                                        mutation_scale = 2)
                patch.remove()
                ax1.add_patch(p_bbox)
            
                
    elif stability_type == "cluster":
        df_stab = df_stab.transpose()
        df_stab.columns +=  1
        
        plt.figure(figsize = (5,8))
        ax1 = sns.boxplot(data = df_stab,
                          boxprops=dict(alpha=.6),
                          linewidth = 2)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel('Jaccard similarity coefficient')
        for patch in reversed(ax1.patches):
            bb = patch.get_path().get_extents()
            color = patch.get_facecolor()
            p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                    abs(bb.width), abs(bb.height),
                                    boxstyle="round,pad=0,rounding_size=.15",
                                    ec="black", fc=color,
                                    mutation_aspect = .05)
            patch.remove()
            ax1.add_patch(p_bbox)
        ax1.set(ylim=(0, 1))
        plt.suptitle(title)
        plt.tight_layout()
            
    if save:
        plt.savefig(f"figures/{savename}.{filetype}")