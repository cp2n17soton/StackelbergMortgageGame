
# =============================================================================
###--- Visuailise Results  ---###
# =============================================================================

##-----------------------------------##
##----- import the dependencies -----##
##-----------------------------------##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from random import shuffle
import datetime as dt
from scipy.stats import skewnorm
import pickle
import collections 
from matplotlib.ticker import FormatStrFormatter
import dill

# =============================================================================
# random seed for reproducibility
# =============================================================================

np.random.seed(seed=8)
random.seed(a=8)
      
# -----------------------------------------------------------------------------
# Switch On-Off
# -----------------------------------------------------------------------------

## Enable program about loading session. 
enb_load_session = True 

## Enble folder directory of picture.
enb_folder_path = True

## Enable visualisation.
enb_vis_plot = True

## Enable saving picture.
enb_save_pic = False

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

# =============================================================================
# Save python session after finish simulation.  
# =============================================================================
## define path and file name to be saved.            
folder_path = "../12. Results/DELL_mgt_results/mgt_0415_02/session/"
filename = folder_path + "mgt_game_0415_02_session.pkl"    

# -----------------------------------------------------------------------------
# load the session using dill
# -----------------------------------------------------------------------------
if enb_load_session == True:
    dill.load_session(filename)
    print("-------------------------------------------------------")
    print("load session : "+str(filename))
    print("-------------------------------------------------------")

# -----------------------------------------------------------------------------
# Setting Variables.
# -----------------------------------------------------------------------------
fig_number = 1
pic_filepath = "../pic/"

## define directory to save picture.     
if enb_folder_path == True:
    pic_filepath = "../pic/"
else:
    pic_filepath = ""

# -----------------------------------------------------------------------------
# Function to plot graph.
# -----------------------------------------------------------------------------

# -------------------------------------------------------------------------
def plot_env_dstb(fig_number, datpck, enb_save_pic):
    ## For graph of distributions of borrower's attributes. 
    env_income_dstb = datpck.env_income_dstb
    env_sample_income_dstb = datpck.env_sample_income_dstb
    env_saving_coef_arr = datpck.env_saving_coef_arr
    env_repay_capabil_arr = datpck.env_repay_capabil_arr
    env_max_loan_term_arr = datpck.env_max_loan_term_arr
    env_eff_growth_rate_arr = datpck.env_eff_growth_rate_arr
 
    # =============================================================================
    ## Plot the histogram of the income distribution.
    # =============================================================================
    n_bins=20
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_income_dstb, bins=n_bins, alpha=0.8)
    plt.title("Annual Income Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'income_dist.png', format='png')
    plt.show() 
    
    fig_number = fig_number + 1
    
    ## =============================================================================
    ### Plot the histogram of sample income distribution.
    ## =============================================================================
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_sample_income_dstb, bins=n_bins, alpha=0.8)
    plt.title("Sampled Annual Income Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'sample_income_dist.png', format='png')
    plt.show() 
    
    fig_number = fig_number + 1
    
    # =============================================================================
    # Plot the distributions.
    # =============================================================================
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_saving_coef_arr , bins=n_bins, alpha=0.8)
    plt.title("Saving Coefficient Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'sample_saving_coef_dist.png', format='png')
    plt.show() 
    fig_number = fig_number + 1
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_repay_capabil_arr , bins=n_bins, alpha=0.8)
    plt.title("Repayment Capability Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'sample_repay_capabil_dist.png', format='png')
    plt.show()
    fig_number = fig_number + 1 
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_max_loan_term_arr , bins=n_bins, alpha=0.8)
    plt.title("Max Loan Term Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'sample_max_loan_term_dist.png', format='png')
    plt.show() 
    fig_number = fig_number + 1
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.hist(env_eff_growth_rate_arr , bins=n_bins, alpha=0.8)
    plt.title("Effective Growth Rate Histogram", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'sample_eff_growth_rate_dist.png', format='png')
    plt.show() 
    fig_number = fig_number + 1

# -------------------------------------------------------------------------     
def plot_bkst_res(fig_number, datpck, enb_save_pic): 
    ## For graph of BankSector improvement over generation.
    bkst_gen_best_idv_df =  datpck.bkst_gen_best_idv_df
    
    ## For graph of the utility over different interest rate.
    df_bkst_Int_rate_res =  datpck.df_bkst_Int_rate_res

    # =====================================================================
    # Plot graph of the improvement over generation.
    # =====================================================================   
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.plot(bkst_gen_best_idv_df["generation"], bkst_gen_best_idv_df["utility"]
            , 'bo-', linewidth=2, markersize=8)
    plt.title("Bank Sector utility over generation", fontsize="24")
    
    plt.xlabel("Generation", fontsize="20")
    plt.xlim(0,max(bkst_gen_best_idv_df["generation"]))
    plt.xticks(fontsize=16)
    
    plt.ylabel("Utility", fontsize="20")
    ylim_offset = max(bkst_gen_best_idv_df["utility"])*0.01
    plt.ylim(min(bkst_gen_best_idv_df["utility"])-ylim_offset
             ,max(bkst_gen_best_idv_df["utility"])+ylim_offset)
    plt.yticks(fontsize=16) 
        
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'bkst_util_improve_gen.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1
    # ---------------------------------------------------------------------    

    # =====================================================================
    # Plot graph of the utility over different interest rate.
    # =====================================================================
    
    ## Sorting value in the DataFrame.
    df_bkst_Int_rate_res = df_bkst_Int_rate_res.sort_values(by=["bkst_Int_rate"])
    
    ## This is the response under the best regulatory LTV ratio.
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.plot(df_bkst_Int_rate_res["bkst_Int_rate"], df_bkst_Int_rate_res["bkst_util_val"]
            , 'bo-', linewidth=2, markersize=5)
    plt.title("Bank Sector utility over Interest rate", fontsize="24")
    
    plt.xlabel("Interest rate", fontsize="20")
    xlim_offset = max(df_bkst_Int_rate_res["bkst_Int_rate"])*0.01
    plt.xlim(min(df_bkst_Int_rate_res["bkst_Int_rate"])-xlim_offset
             ,max(df_bkst_Int_rate_res["bkst_Int_rate"])+xlim_offset)
    plt.xticks(fontsize=16)
    
    plt.ylabel("Utility", fontsize="20")
    ylim_offset = max(df_bkst_Int_rate_res["bkst_util_val"])*0.02
    plt.ylim(min(df_bkst_Int_rate_res["bkst_util_val"])-ylim_offset
             ,max(df_bkst_Int_rate_res["bkst_util_val"])+ylim_offset)
    plt.yticks(fontsize=16) 
        
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'bkst_util_on_Int_rate.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1 
    # ---------------------------------------------------------------------
    # Plot BankSector utility/profit, loan amount, and total repayment.
    # ---------------------------------------------------------------------
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Bank Loan and Profit over Interest rate", fontsize="24")
    
    plt.plot(df_bkst_Int_rate_res["bkst_Int_rate"], df_bkst_Int_rate_res["bkst_util_val"]
            , 'o-', linewidth=2, markersize=5)
    plt.plot(df_bkst_Int_rate_res["bkst_Int_rate"], df_bkst_Int_rate_res["loan_amt"]
            , 'o-', linewidth=2, markersize=5)
    plt.plot(df_bkst_Int_rate_res["bkst_Int_rate"], df_bkst_Int_rate_res["total_repay_amt"]
            , 'o-', linewidth=2, markersize=5)
    
    plt.xlabel("Interest rate", fontsize="20")
    plt.ylabel("Value", fontsize="20")
    
    plt.legend(["Profit/Utility","Loan Amount","Total Repayment"]
            , fontsize=14)
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'bkst_util_loan_repay_on_Int_rate.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1 
    
 # -------------------------------------------------------------------------   
def plot_regt_res(fig_number, datpck, enb_save_pic): 
    ## For graph of Regulator utility over different LTV ratios.
    df_regt_D_idv_list =  datpck.df_regt_D_idv_list
    
    ## For graph of the n_brw_NotBuy over LTV ratios.
    df_regt_LTV_res = datpck.df_regt_LTV_res
    # ---------------------------------------------------------------------
   
    # =============================================================================
    # Plot graph of the utility over regulatory LTV ratio.
    # =============================================================================
    
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.plot(df_regt_D_idv_list["Reg_LTV"], df_regt_D_idv_list["utility"]
            , 'bo-', linewidth=2, markersize=8)
    plt.title("Regulator utility over LTV ratio", fontsize="24")
    
    plt.xlabel("Regulatory LTV ratio", fontsize="20")
    xlim_offset = max(df_regt_D_idv_list["Reg_LTV"])*0.01
    plt.xlim(min(df_regt_D_idv_list["Reg_LTV"])-xlim_offset
             ,max(df_regt_D_idv_list["Reg_LTV"])+xlim_offset)
    plt.xticks(fontsize=16)
    
    plt.ylabel("Utility", fontsize="20")
    ylim_offset = max(df_regt_D_idv_list["utility"])*0.02
    plt.ylim(min(df_regt_D_idv_list["utility"])-ylim_offset
             ,max(df_regt_D_idv_list["utility"])+ylim_offset)
    plt.yticks(fontsize=16) 
        
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'regt_util_on_LTV_ratio.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1    
    
    # -----------------------------------------------------------------------------
    # Plot graph of Good welfare over regulatory LTV ratios.
    # -----------------------------------------------------------------------------
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("Good social welfare over LTV ratios", fontsize="24")
    
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["welfare_Good"]
            , 'o-', linewidth=2, markersize=5)
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["brw_util_Good"]
            , 'o-', linewidth=2, markersize=5)
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["bank_profit_Good"]
            , 'o-', linewidth=2, markersize=5)
    
    xlim_offset = max(df_regt_LTV_res["max_LTV_ratio"])*0.01
    plt.xlim(min(df_regt_LTV_res["max_LTV_ratio"])-xlim_offset
             ,max(df_regt_LTV_res["max_LTV_ratio"])+xlim_offset)
    
    plt.legend(["Good Social Welfare","Good Borrower Utility","Good Bank Profit"]
            , fontsize=14)

    plt.xlabel("Regulatory LTV ratio", fontsize="20")
    plt.ylabel("Value", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'regt_good_util_on_LTV_ratio.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1 
    # -----------------------------------------------------------------------------
    
    # -----------------------------------------------------------------------------
    # Plot graph of n_brw_NotBuy over regulatory LTV ratios.
    # -----------------------------------------------------------------------------
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title("People that did not buy a house over LTV ratios", fontsize="24")
    
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["n_brw"]
            , 'o-', linewidth=2, markersize=5)
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["n_brw_NotBuy"]
            , 'o-', linewidth=2, markersize=3)
    plt.plot(df_regt_LTV_res["max_LTV_ratio"], df_regt_LTV_res["n_brw_NotBorrow"]
            , 'x', linewidth=2, markersize=5)
    
    xlim_offset = max(df_regt_LTV_res["max_LTV_ratio"])*0.01
    plt.xlim(min(df_regt_LTV_res["max_LTV_ratio"])-xlim_offset
             ,max(df_regt_LTV_res["max_LTV_ratio"])+xlim_offset)
    
    plt.legend(["All People","Not Buy","Not Borrow"]
            , fontsize=14)

    plt.xlabel("Regulatory LTV ratio", fontsize="20")
    plt.ylabel("Persons", fontsize="20")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16) 
    plt.grid()
    ax.set_axisbelow(True)
    if enb_save_pic == True:
        plt.savefig(pic_filepath+'regt_brwNotBuy_on_LTV_ratio.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1 

# ------------------------------------------------------------------------- 
    
def plot_graphs(fig_number, datpck, enb_save_pic):
    ## Plot graphs of distributions
    ## and plot graphs of BankSector results 
    ## and graphs of Regulator results.
    plot_env_dstb(fig_number, datpck, enb_save_pic)
    plot_bkst_res(fig_number, datpck, enb_save_pic)
    plot_regt_res(fig_number, datpck, enb_save_pic)

# -------------------------------------------------------------------------    
# Boxplot visualisation.    
# -------------------------------------------------------------------------

def boxplot_multisim_res(fig_number, df_x, df_y
                          , plt_title, plt_xlabel, plt_ylabel
                          , bp_ylegend): 
    ## Prep data for boxplot function.
    xval = df_x.iloc[0]
    df_same_xval = df_x[df_x==xval]
    n_rnd = len(df_same_xval)
    n_box = int(np.around( len(df_x) / n_rnd ))
    
    data_lst = []
    med_lst = []
    for bx in range(n_box):
        val_lst = []
        for nn in range(n_rnd):
            idx = (bx*n_rnd) + nn
            val = df_y.iloc[idx]
            val_lst.append(val)
        data = np.array(val_lst)
        med = np.median(data)
        ## add data into list.
        data_lst.append(data)
        med_lst.append(med)
    
    ## Prep data for axis.
    pos_lst = list(df_x.drop_duplicates())
    
    ## Prep widths of boxplot
    width = 0.4*(pos_lst[1] - pos_lst[0])
    width = np.around(width, decimals=4)
    print("width = "+str(width))
    
    ## Prep limit of x-axis
    xaxis_min = min(pos_lst) - width
    xaxis_max = max(pos_lst) + width
    
    # --------------------------------------------------------------------- 
    # Plot graph 
    # ---------------------------------------------------------------------
    fig = plt.figure(fig_number)
    fig, ax = plt.subplots(figsize=(10, 5))

    ## line graph
    plt.plot(pos_lst, med_lst, '--o'
             , label=bp_ylegend )
    
    ## boxplot
    bp = ax.boxplot(x=data_lst, positions=pos_lst
                , notch=0, vert=1, whis=1.5 , widths=width)
    
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black', marker='.')
    plt.setp(bp['medians'], color='red')
    
    plt.title(plt_title, fontsize="20")
    
    plt.xlabel(plt_xlabel, fontsize="20")
    plt.ylabel(plt_ylabel, fontsize="20")

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)     
    
    plt.xlim(xaxis_min,xaxis_max)
            
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    
    plt.grid()
    ax.set_axisbelow(True)
    plt.legend()
    plt.savefig(pic_filepath + plt_title + '.png', format='png')
    plt.show()
    
    fig_number = fig_number + 1    

# ------------------------------------------------------------------------- 
def three_boxplots(fig_number, df_x, plt_xlabel
                   , title_suffix, df_out_scen_lst):
    
    ## result of the optimal regulatory LTV ratio 
    df_y = df_out_scen_lst["regt_best_Reg_LTV"]
    plt_title = "Optimal LTV ratio " + title_suffix
    
    plt_ylabel = "Optimal LTV ratio"
    bp_ylegend = "median value"
    boxplot_multisim_res(fig_number, df_x, df_y
                          , plt_title, plt_xlabel, plt_ylabel
                          , bp_ylegend)
    
    ## result of the optimal interest rate
    df_y = df_out_scen_lst["bkst_best_Int_rate"]
    plt_title = "Optimal interest rate " + title_suffix
    plt_ylabel = "Optimal Interest Rate"
    bp_ylegend = "median value" 
    boxplot_multisim_res(fig_number, df_x, df_y
                          , plt_title, plt_xlabel, plt_ylabel
                          , bp_ylegend)
    
    ## result of the people that cannot buy a house
    df_y = 100*(df_out_scen_lst["n_brw_NotBuy"]/df_out_scen_lst["n_brw"])
    plt_title = "People who did not buy a house " + title_suffix
    plt_ylabel = "% of people"
    bp_ylegend = "median value" 
    boxplot_multisim_res(fig_number, df_x, df_y
                          , plt_title, plt_xlabel, plt_ylabel
                          , bp_ylegend)
    
# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
    
# =============================================================================
# Visualisations
# =============================================================================
    
if enb_vis_plot == True:
    ## set the index value to get data from the list.
    idx = 202
# -----------------------------------------------------------------------------
# Plot distributions and Bank results and Regulator results.
# -----------------------------------------------------------------------------
    datpck = datpck_lst_6[idx]
    plot_graphs(fig_number, datpck, enb_save_pic)   
    
# -----------------------------------------------------------------------------
# Plot boxplot of the top-level results.
# -----------------------------------------------------------------------------
    df_out_scen_lst = df_out_scen_lst_6
    df_x = df_out_scen_lst["LTI_limit"]
    plt_xlabel = "LTI limit"
    title_suffix = "over LTI limit"
        
    three_boxplots(fig_number, df_x, plt_xlabel
            , title_suffix, df_out_scen_lst)
    
# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
    
    

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
    
    

# =============================================================================
# 
# =============================================================================



