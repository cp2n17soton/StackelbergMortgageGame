
# =============================================================================
###--- Simulation model implementation  ---###
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

# =============================================================================
# Class of the Borrower
# =============================================================================

class Borrower(object):
    """__init__() functions as the class constructor"""
    def __init__(self, brw_id=None, ann_income=None, saving_money=None, 
                 repay_capabil=None, max_loan_term=None, eff_growth_rate=None):
        # identity 
        self.brw_id = brw_id
        
        # profile
        self.ann_income = ann_income
        self.saving_money = saving_money
        self.repay_capabil = repay_capabil 
        self.max_loan_term = max_loan_term
        
        # house profile
        ## effective growth rate of house price ##
        self.eff_growth_rate = eff_growth_rate 
                
        ## chosen bank decision variable
        self.bank_id = None
        self.interest_rate = None
        
        # borrowing decision variable
        self.borrow_D = None
        
        # original decision variable
        self.buying_D = None
        self.buy_price = None           
        self.LTV_ratio = None
        self.loan_term = None
        
        # utility value
        self.util_val = None
 
        # results from the decision
        self.LTI_ratio = None
        self.monthly_repay_amt = None
        self.total_repay_amt = None
        self.loan_amt = None
        self.bank_profit = None
        
       
    def monthly_repay_cal(self, loan_amount, loan_term, mthly_interest_rate):   
        if loan_amount < 0:
            #print("loan_amount < 0")
            loan_amount = 0
        
        if loan_term <= 0:
            #print("loan_term <= 0")
            loan_term = 1
        
        mg = np.power((1+mthly_interest_rate),loan_term)        
        if mthly_interest_rate == 0:
            repay = loan_amount/loan_term
        else:
            repay = (mthly_interest_rate*loan_amount*mg)/(mg-1)        
        return repay
    
    def utility_cal(self, price, LTV_ratio, eff_growth_rate, loan_term, interest_rate):
        loan_amount = LTV_ratio*price - self.saving_money
        mthly_interest_rate = interest_rate/12
        
        mthly_repay = self.monthly_repay_cal(loan_amount, loan_term, mthly_interest_rate)
        
        year_loan_term = loan_term/12
        val_growth = np.power((1 + eff_growth_rate),year_loan_term)
        
        house_using_value = price
        
        util_val = price*val_growth - (mthly_repay*loan_term) + house_using_value
        util_val = np.around(util_val)
        return util_val
    
    def Check_Pimax_Pmin(self, max_LTV_ratio, interest_rate, min_house_price):
        if max_LTV_ratio >= 1:
            #print("max_LTV_ratio >= 1")
            max_LTV_ratio = 0.9999
            
        P_max_1 = self.saving_money / (1 - max_LTV_ratio)
        
        P_max_2 = self.Cal_Price_max_Repay_cap(self.max_loan_term, interest_rate
                                , self.ann_income, self.saving_money
                                , self.repay_capabil)
        
        ## P_max is the upper bound of the possible house price.
        P_max = P_max_1
        if P_max_2 < P_max_1:
            P_max = P_max_2
            
        result = False
        if P_max >= min_house_price:
            result = True
        else:
            result = False
            
        return result
    
    def Cal_Price_max_Repay_cap(self, loan_term, interest_rate
                                , ann_income, saving_money
                                , repay_capabil):
        ## Calculate the maximum house price (upper bound) based on the
        ##### constraint of the monthly repayment capability.
        mthly_interest_rate = interest_rate/12
        mthly_income = ann_income / 12
        
        if mthly_interest_rate == 0:
            Loan_amt_max = ( repay_capabil*mthly_income )*loan_term
        else:
            mg = np.power((1+mthly_interest_rate),loan_term) 
            Loan_amt_max = ( repay_capabil*mthly_income )*( (mg-1)/(mthly_interest_rate*mg) )
            
        Price_max = np.around( Loan_amt_max + saving_money )
        
        return Price_max
    
    def Check_repay_Pmin(self, interest_rate, min_house_price):
        loan_min = min_house_price - self.saving_money
        mthly_interest_rate = interest_rate/12
        m_i = self.ann_income / 12
        
        if loan_min < 0 :
            loan_min = 0
        
        c_i_min = self.monthly_repay_cal(loan_min, self.max_loan_term, mthly_interest_rate)
        
        if m_i ==0:
            print("m_i = 0")
            m_i = 0.0001
            
        repay_burden_min = c_i_min / m_i
        result = False
        if repay_burden_min <= self.repay_capabil:
            result = True
        else:
            result = False
            
        return result
        
    def Buying_decision(self, max_LTV_ratio, interest_rate, min_house_price):
        
        buy = False
        
        if max_LTV_ratio > 1:
            print("Invalid maximum LTV ratio : max_LTV_ratio = "+str(max_LTV_ratio))
        
        res1 = self.Check_Pimax_Pmin(max_LTV_ratio, interest_rate, min_house_price)
        res2 = self.Check_repay_Pmin(interest_rate, min_house_price)
        
        if (res1 and res2) == True:
            buy = True
        else:
            buy = False
        
        return buy
    
    def setBuying_decision(self, buying_D, buy_price, loan_term
                           , intr_rate, growth_rate):
        self.buying_D = buying_D
        self.buy_price = np.around(buy_price, decimals=0)
        
        
        ## Set borrowing decision.
        self.borrow_D = bool(buy_price > self.saving_money)
        
        if self.borrow_D == True:
            ## Set the LTV ratio value based on the decided house Price.
            LTV_ratio = (buy_price - self.saving_money) / buy_price
            
            # Set the monthly_repay_amt
            mthly_intr_rate = intr_rate/12
            loan_amount = np.around(buy_price - self.saving_money)
            monthly_repay_amt = self.monthly_repay_cal(loan_amount
                                            , loan_term, mthly_intr_rate)
            total_repay_amt = monthly_repay_amt * loan_term
            LTI_ratio = loan_amount/ self.ann_income
            self.loan_term = loan_term
            self.loan_amt = loan_amount
            
            util_val = self.utility_cal(buy_price, LTV_ratio
                        , growth_rate, loan_term, intr_rate) 
            bank_profit = total_repay_amt - loan_amount
        else:
            LTV_ratio = 0
            monthly_repay_amt = 0
            total_repay_amt = 0
            LTI_ratio = 0
            self.loan_term = 0
            self.loan_amt = 0
            util_val = 0
            bank_profit = 0
        
        
        self.monthly_repay_amt = np.around(monthly_repay_amt, decimals=2)
        self.total_repay_amt = np.around(total_repay_amt, decimals=0)
        self.LTV_ratio = np.around(LTV_ratio, decimals=4)
        self.LTI_ratio = np.around(LTI_ratio, decimals=4)
        
        # Set the utility value of the decision.
        self.util_val = util_val
        
        ## Bank profit for this loan.
        self.bank_profit = np.around(bank_profit)
 
# =============================================================================
#  Class of Commerical Bank        
# =============================================================================
class Bank(object):
    def __init__(self, bank_id=None):
        self.bank_id = bank_id
        
        ## profile - constraints
        #self.max_num_brw = None
        #self.total_loan_amt_limit = None
        
        # decision variable
        self.interest_rate = None
        
        # utility value
        self.util_val = None
        
        # list of borrower information
        self.brw_id_lst = []
        self.total_repay_amt_lst = []
        self.loan_amt_lst = []
        self.profit_lst = []
        
        # results of decision
        self.num_brw = None
        
# =============================================================================
# Class of Banking Sector
# =============================================================================
class BankSector(object):
    def __init__(self, n_bank):
        self.n_bank = n_bank
        
        # list of Bank object
        self.bank_lst = None
        
        # decision variable
        self.interest_rate = 0.08   ## default value of interest rate.
    
        # utility value
        self.util_val = None
        
    def init_BankList(self):
        bank_lst = []
        for i in range(self.n_bank):
            bank_id = "B" + str(i+1).rjust(2,'0')
            bank = Bank(bank_id)
            
            ## set the interest rate as the same rate.
            bank.interest_rate = self.interest_rate
            
            bank_lst.append(bank)
        self.bank_lst = bank_lst
        
    def brw_choose_bank(self, brw_lst):
        for brw in brw_lst:
            chosen_idx = np.random.randint(0,self.n_bank)
            brw.bank_id = self.bank_lst[chosen_idx].bank_id
            brw.interest_rate = self.bank_lst[chosen_idx].interest_rate
            
            self.bank_lst[chosen_idx].brw_id_lst.append(brw.brw_id)
    
    def Collect_brw_D(self, brw_lst):
        sum_profit = 0
        for bank in self.bank_lst:
            for brw_id in bank.brw_id_lst:
                brw_idx = brw_id - 1
                brw = brw_lst[brw_idx]
                
                bank.total_repay_amt_lst.append(brw.total_repay_amt)
                bank.loan_amt_lst.append(brw.loan_amt)
                
                profit = brw.bank_profit
                bank.profit_lst.append(profit)
        
            bank.num_brw = len(bank.brw_id_lst)
            bank.util_val = sum(bank.profit_lst)
            ## sum of Bank's profit.
            sum_profit = sum_profit + bank.util_val
            
        ## set the utility value of BankSector from the sum. 
        self.util_val = sum_profit
    
    def Cal_bkst_util(self, brw_lst, Int_rate
                    , min_house_price, max_LTV_ratio
                    , Cal_brw_Price_Term):
        ## get the interest rate. 
        self.interest_rate = Int_rate
        
        ## create a new bank list from the interest rate.
        self.init_BankList()
        self.brw_choose_bank(brw_lst)

        #optmz = Optimizer()
        # -----------------------------------------------------------------
        # Calculate the Price and Term for each Borrower.
        # -----------------------------------------------------------------
        for idx in range(len(brw_lst)):
            brw = brw_lst[idx]
            
            brw_buy_D = brw.Buying_decision(max_LTV_ratio
                                    , brw.interest_rate, min_house_price)
            brw.buying_D = brw_buy_D
            
            Price, Term, LTV_ratio = Cal_brw_Price_Term(max_LTV_ratio
                                , brw.interest_rate, min_house_price
                                , brw.eff_growth_rate, brw.ann_income
                                , brw.saving_money, brw.repay_capabil
                                , brw.max_loan_term, brw_buy_D)
            
            
            brw.setBuying_decision(brw_buy_D, Price, Term
                                   , brw.interest_rate
                                   , brw.eff_growth_rate)
            
        # -----------------------------------------------------------------
        # Collect the data for each Borrower into the List in BankSector.
        # -----------------------------------------------------------------
        self.Collect_brw_D(brw_lst)   
        # -----------------------------------------------------------------
        ## output of this function is utility value of BankSector. 
        return self.util_val
            
# =============================================================================
# Class of Regulator    
# =============================================================================
class Regulator(object):
    def __init__(self, regr_id=None, w_alpha=None, w_beta=None
                 , LTI_limit=None) :
        self.regr_id = regr_id
        
        # policy variable
        self.w_alpha = w_alpha
        self.w_beta = w_beta
        self.LTI_limit = LTI_limit
        
        # decision variable
        self.max_LTV_ratio = None
        
         ## DataFrame of simulated results.
        self.df_brw_lst = None
        
        # utility value
        self.util_val = None
        self.welfare_Good = None
        self.welfare_Bad = None
        
        ## result of decided policy.
        self.n_brw_Good = None
        self.n_brw_Bad = None
        
        self.brw_util_Good = None
        self.brw_util_Bad = None
        
        self.bank_profit_Good = None
        self.bank_profit_Bad = None
        
        self.n_brw_NotBuy = None
        self.n_brw_NotBorrow = None
        self.n_brw = None
        
        ## BankSector optimization data.
        self.bkst_DE_output = None
        
    def Cal_regt_util(self, optmz, min_house_price
                      , Reg_LTV_ratio, bkst, brw_lst):
        ## Optimize to get the best interest rate.
        self.bkst_DE_output = optmz.BankSector_DE(min_house_price  
                      , Reg_LTV_ratio
                      , bkst, brw_lst)
        ## BankSector optimal interest rate.
        bkst_best_Int_rate = self.bkst_DE_output[0]

        bkst.Cal_bkst_util(brw_lst, bkst_best_Int_rate
                    , min_house_price, Reg_LTV_ratio
                    , optmz.Cal_brw_Price_Term)

        df_brw_lst = df_from_obj_list(brw_lst)

        self.df_brw_lst = df_brw_lst 

        self.max_LTV_ratio = Reg_LTV_ratio
        #print("regt self.max_LTV_ratio = "+str(self.max_LTV_ratio))

        LTI_limit = self.LTI_limit

        ## get the records of Good loans and Bad loans.
        df_brw_lst_Good = df_brw_lst[df_brw_lst.LTI_ratio < LTI_limit ]
        df_brw_lst_Bad = df_brw_lst[df_brw_lst.LTI_ratio >= LTI_limit ]
        
        ## record the number of Good loans and Bad loans.
        self.n_brw_Good = len(df_brw_lst_Good)
        self.n_brw_Bad = len(df_brw_lst_Bad) 

        ## calculate the utility of banks and borrowers.
        bank_profit_Good = sum(df_brw_lst_Good["bank_profit"])    
        brw_util_Good = sum(df_brw_lst_Good["util_val"])

        bank_profit_Bad = sum(df_brw_lst_Bad["bank_profit"])    
        brw_util_Bad = sum(df_brw_lst_Bad["util_val"])

        ## record the utility of banks and borrowers.
        self.brw_util_Good = brw_util_Good
        self.brw_util_Bad = brw_util_Bad

        self.bank_profit_Good = bank_profit_Good
        self.bank_profit_Bad = bank_profit_Bad

        ## calculate and record the social welfare.
        self.welfare_Good = bank_profit_Good + brw_util_Good 
        self.welfare_Bad = bank_profit_Bad + brw_util_Bad

        ## calculate and record the utility of Regulator.
        util_value = self.w_alpha*self.welfare_Good - self.w_beta*self.welfare_Bad
        util_value = np.around(util_value, decimals=2)   
        
        self.util_val = util_value
        #print("regt self.util_val = "+str(self.util_val))
        
        self.n_brw_NotBuy = len(df_brw_lst[df_brw_lst["buying_D"]==False])
        self.n_brw_NotBorrow = len(df_brw_lst[df_brw_lst["borrow_D"]==False])
        self.n_brw = len(df_brw_lst)

        return util_value
        
# =============================================================================
# function of creating a value from normal distribution.
# =============================================================================

def Cal_norm_val_arr(n_val, mean_val, sigma, decimal, low_bnd, upp_bnd):
    nml_arr = np.random.normal(mean_val, sigma, n_val)
    for i in range(n_val):    
        val = nml_arr[i]
        if val < low_bnd:
            val = low_bnd
        if val > upp_bnd:
            val = upp_bnd
        val = np.around(val, decimals=decimal)
        nml_arr[i] = val
    return nml_arr

# =============================================================================
# Class of Environment    
# =============================================================================
        
class Environment(object):
    def __init__(self):
        ## population variables.
        self.n_dstb_sample = 1000
        self.n_brw = 10
        self.n_bank = 10
               
        ## variable to keep the object of agents.
        self.brw_lst = None
        self.bkst = None
        self.regt = None
        
        ## variable to keep the distributions.
        self.income_dstb = None
        self.sample_income_dstb = []
        
        # median annual income in this scenario.
        self.median_ann_income = 30000
         
        # ratio represent the expensiveness of the house price 
        # Price-to-Income ratio
        self.min_Price_PTI_ratio = 2.5   ## 1st baseline 2.5 ##
        
        # minimum house price
        self.min_house_price = None
         
        ## mean value of the normal distributions.
        self.mean_saving_coef = 0.4     ## 1st baseline 0.4 ##
        self.mean_repay_capabil = 0.4   ## 1st baseline 0.4 ##
        self.mean_max_loan_term = 300   ## 1st baseline 300 ##
        self.mean_eff_growth_rate = 0.05  ## 1st baseline 0.05 ##
         
        ## array that contain the data from the distributions.
        self.saving_coef_arr = None
        self.repay_capabil_arr = None
        self.max_loan_term_arr = None
        self.eff_growth_rate_arr = None
 
         
    ## Manually set the income distribution.
    def set_income_distribution(self,n_sample):
         inc_lst = [10000,18000,20000,22000,24000,30000,34000,40000,48000,68000]
         income_val = np.array(inc_lst)
         p = [0.04,0.14,0.15,0.19,0.17,0.14,0.12,0.03,0.015,0.005]
         rnd_income = np.random.choice(income_val, n_sample, p)
         
         self.income_dstb = rnd_income
    
    ## Generate the income distribution from skewnorm function.
    def set_ann_income_dstb(self, n_sample):
        ## Set the parameter value of the function.
        alp_val = 4 # parameter of skewness
        loc_val = 1.0 # location
        scl_val = 1.0 # scale
        rsk_dstb = skewnorm.rvs(a=alp_val, loc=loc_val, scale=scl_val, size=n_sample)
        
        income_dstb_rvs = (rsk_dstb*self.median_ann_income) / np.median(rsk_dstb)
        income_dstb_rvs = np.around(income_dstb_rvs)
    
        self.income_dstb =  income_dstb_rvs
        
        
    def set_brw_var_dstb(self, population):
        ## upper and lower bound of Borrower number.
        if population < 1:
            population = 1
        elif population > self.n_dstb_sample:
            population = self.n_dstb_sample
        
        mean_val = self.mean_saving_coef # mean and standard deviation
        sigma , decimal_place = 0.15 , 4  
        lower_bound, upper_bound = 0, 2
        self.saving_coef_arr = Cal_norm_val_arr(population, mean_val, sigma
                            , decimal_place, lower_bound, upper_bound)
        
        mean_val = self.mean_repay_capabil
        sigma , decimal_place = 0.1 , 4
        lower_bound, upper_bound = 0.1, 0.7
        self.repay_capabil_arr = Cal_norm_val_arr(population, mean_val, sigma
                            , decimal_place, lower_bound, upper_bound)
        
        mean_val = self.mean_max_loan_term
        sigma , decimal_place = 60 , 0
        lower_bound, upper_bound = 1, 480
        self.max_loan_term_arr = Cal_norm_val_arr(population, mean_val, sigma
                            , decimal_place, lower_bound, upper_bound)
        
        mean_val = self.mean_eff_growth_rate
        sigma , decimal_place = 0.01 , 4
        lower_bound, upper_bound = 0.02, 0.50
        self.eff_growth_rate_arr = Cal_norm_val_arr(population, mean_val, sigma
                            , decimal_place, lower_bound, upper_bound)
        
        
    def set_min_house_price(self):
        self.min_house_price = self.min_Price_PTI_ratio*np.median(self.income_dstb)   
        
        ## use the sample income distribution instead.
        ##self.min_house_price = self.min_Price_PTI_ratio*np.median(self.sample_income_dstb) 
         
    def init_borrowerList(self,population): 
        ## upper and lower bound of Borrower number.
        if population < 1:
            population = 1
        elif population > self.n_dstb_sample:
            population = self.n_dstb_sample
            
        ## create and set the distribution.
        self.set_brw_var_dstb(population)
        
        lst = []
        for i in range(population):            
            annual_income = self.income_dstb[i]
            
            saving_coef = self.saving_coef_arr[i]
            
            
            saving_money = np.around( saving_coef*annual_income )
            
            repay_capability = self.repay_capabil_arr[i]
            
            
            max_loan_term = self.max_loan_term_arr[i]
            
            
            eff_growth_rate = self.eff_growth_rate_arr[i]
            
            
            brw = Borrower(i+1, annual_income, saving_money,
                           repay_capability, max_loan_term, eff_growth_rate)
            lst.append(brw)
            
            ## keep the sample income.
            self.sample_income_dstb.append(annual_income)
        self.brw_lst = lst
                       
# =============================================================================
# Function for transform the Object List into the DataFrame.
# =============================================================================

def df_from_obj(obj):
    keys = obj.__dict__.keys()
    col_name_lst = list(keys)
    
    records_lst = []
    values = obj.__dict__.values()
    row_value_lst = list(values)
    records_lst.append(row_value_lst)    
    
    df = pd.DataFrame(records_lst, columns = col_name_lst)
    return df   

def df_from_obj_list(obj_list):
    # get attribute to be column name.
    obj = obj_list[0]
    keys = obj.__dict__.keys()
    col_name_lst = list(keys)
        
    records_lst = []
    for obj in obj_list:
        values = obj.__dict__.values()
        row_value_lst = list(values)
        records_lst.append(row_value_lst)
        
    df = pd.DataFrame(records_lst, columns = col_name_lst)
    return df

# =============================================================================
# Function to create a DataFrame from list of column name and data.
# =============================================================================

def df_from_col_list(col_names, col_data_lst):
    n_col = len(col_names)
    prep_data = []
    for ii_col in range(n_col):
        col = []
        col.append(col_names[ii_col])
        col.append(col_data_lst[ii_col])
        prep_data.append(col)
        
    df = pd.DataFrame.from_dict(collections.OrderedDict(prep_data))
    return df

# =============================================================================
# Class of Data Collector
# =============================================================================

class DataCollector(object):
    def __init__(self):
        self.is_bkst_enable = False
        self.is_regt_enable = False

        ## result of each interest rate of BankSector.
        self.bkst_Int_rate_res = None
        
        ## result of each max LTV ratio of Regulator
        self.regt_LTV_res = None
        
        
    def clear_atrb_val(self):
        self.bkst_Int_rate_res = None
        
    def Collect_bkst_Int_rate_result(self, bkst):
        if self.is_bkst_enable == True :    
            sum_num_brw = 0
            sum_loan_amt = 0
            sum_total_repay_amt = 0
            for bank in bkst.bank_lst:    
                sum_num_brw = sum_num_brw + bank.num_brw
            
                for loan_amt in bank.loan_amt_lst:
                    sum_loan_amt = sum_loan_amt + loan_amt
            
                for total_repay_amt in bank.total_repay_amt_lst:
                    sum_total_repay_amt = sum_total_repay_amt + total_repay_amt 
        
            col_dat_lst = []
            col_dat_lst.append([bkst.util_val])
            col_dat_lst.append([bkst.interest_rate])
        
            col_dat_lst.append([sum_num_brw])
            col_dat_lst.append([sum_loan_amt])
            col_dat_lst.append([sum_total_repay_amt])
        
        
            col_name = ["bkst_util_val", "bkst_Int_rate"
                        ,"n_brw_Borrow","loan_amt","total_repay_amt"]
        
            df_bkst_res_new = df_from_col_list(col_name ,col_dat_lst)
            
            if self.bkst_Int_rate_res is not None:
                # Append the new Record into the DataFrame.
                df_bkst_res = pd.concat([self.bkst_Int_rate_res, df_bkst_res_new]
                        , ignore_index=True)
            else: 
                df_bkst_res = df_bkst_res_new
                
            self.bkst_Int_rate_res = df_bkst_res
        
    def Collect_regt_LTV_result(self, regt):
        if self.is_regt_enable == True:
            col_dat_lst = []
            
            col_dat_lst.append([regt.n_brw_Good])
            col_dat_lst.append([regt.n_brw_Bad])
            
            col_dat_lst.append([regt.max_LTV_ratio])
            col_dat_lst.append([regt.util_val])
            
            col_dat_lst.append([regt.welfare_Good])
            col_dat_lst.append([regt.welfare_Bad])
            
            col_dat_lst.append([regt.brw_util_Good])
            col_dat_lst.append([regt.brw_util_Bad])
            col_dat_lst.append([regt.bank_profit_Good])
            col_dat_lst.append([regt.bank_profit_Bad])
            
            col_dat_lst.append([regt.w_alpha])
            col_dat_lst.append([regt.w_beta])
            col_dat_lst.append([regt.LTI_limit])
            
            col_dat_lst.append([regt.n_brw_NotBuy])
            col_dat_lst.append([regt.n_brw_NotBorrow])
            col_dat_lst.append([regt.n_brw])
            
            col_name = ['n_brw_Good','n_brw_Bad'
                        ,"max_LTV_ratio","regt_util_val"
                        ,"welfare_Good","welfare_Bad"
                        ,"brw_util_Good","brw_util_Bad"
                        ,"bank_profit_Good","bank_profit_Bad"
                        ,"w_alpha","w_beta","LTI_limit"
                        ,"n_brw_NotBuy","n_brw_NotBorrow"
                        ,'n_brw']
            
            df_new = df_from_col_list(col_name ,col_dat_lst)
            
            if self.regt_LTV_res is not None:
                # Append the new Record into the DataFrame.
                df_regt_res = pd.concat([self.regt_LTV_res, df_new]
                        , ignore_index=True)
            else:
                df_regt_res = df_new
            
            self.regt_LTV_res = df_regt_res
                
# =============================================================================
# Class of Decision individual for the DE algorithm
# =============================================================================

class brw_Decision_idv(object):
    def __init__(self, generation=None, D_idv_id=None
                 , Price=None, Term=None, LTV_ratio=None
                 , utility=None):
        self.generation = generation
        self.D_idv_id = D_idv_id
        self.Price = Price
        self.Term = Term
        self.LTV_ratio = LTV_ratio
        self.utility = None

class bkst_Decision_idv(object):
    def __init__(self, generation=None, D_idv_id=None
                 , Int_rate=None, utility=None): 
        self.generation = generation
        self.D_idv_id = D_idv_id
        self.Int_rate = Int_rate
        self.utility = None

class regt_Decision_idv(object):
    def __init__(self, generation=None, D_idv_id=None
                 , Reg_LTV=None, utility=None): 
        self.generation = generation
        self.D_idv_id = D_idv_id
        self.Reg_LTV = Reg_LTV
        self.utility = None
          
# =============================================================================
# Class of the Optimizer 
# =============================================================================

class Optimizer(object):
    
    def __init__(self, data_collector=None):
        self.clct = data_collector
        
        ## BankSector DE parameters
        self.bkst_ngenf = 10  ## ngenf is number of generation, the iteration round.
        self.bkst_npf = 20 ## npf is number of population, the population size.
        self.bkst_CR = 0.9 ## CR is Crossover Rate.
        self.bkst_Ft = 0.8 ## Scaling Factor.
        self.ir_max = 1.00  ## upper bound of Interest rate
        self.ir_min = 0.00  ## lower bound of Interest rate

        ## regt OneOpt parameters
        self.LTV_max = 1.00  ## upper bound of regulatory LTV ratio
        self.LTV_min = 0.00  ## lower bound of regulatory LTV ratio
 
    ##-----------------------------------------------------------------------##
    ##  Functions in the DE algrorithm for Borrower.
    ##-----------------------------------------------------------------------##         

    def Cal_Price_max_Repay_cap(self, loan_term, interest_rate
                                , ann_income, saving_money
                                , repay_capabil):
        ## Calculate the maximum house price (upper bound) based on the
        ##### constraint of the monthly repayment capability.
        mthly_interest_rate = interest_rate/12
        mthly_income = ann_income / 12
        
        if mthly_interest_rate == 0:
            Loan_amt_max = ( repay_capabil*mthly_income )*loan_term
        else:
            mg = np.power((1+mthly_interest_rate), loan_term) 
            Loan_amt_max = ( repay_capabil*mthly_income )*( (mg-1)/(mthly_interest_rate*mg) )
            
        Price_max = np.around( Loan_amt_max + saving_money )
        
        return Price_max
    
    def Cal_Term_min_Repay_cap(self, P_min, interest_rate
                                , ann_income, saving_money
                                , repay_capabil):
        mthly_interest_rate = interest_rate/12
        mthly_income = ann_income / 12
        L_min = P_min - saving_money
        c_max =  repay_capabil*mthly_income 
        
        if c_max > 0:
            if mthly_interest_rate == 0:
                Term_min = L_min/c_max  
            else:
                up_term = np.log(c_max) - np.log(c_max - (mthly_interest_rate*L_min))
                bt_term = np.log(1+mthly_interest_rate)
                Term_min = up_term/bt_term
        else:
            Term_min = 1
            
        ## lower bound 
        if Term_min < 1:
            Term_min = 1
            
        Term_min = np.around(Term_min)
        return Term_min  
    
    def init_ppl_brw_D_idv(self, population_size
                           , P_min, P_max
                           , max_loan_term 
                           , interest_rate, saving_money 
                           , ann_income, repay_capabil):
        
        ## initiate the population of the individual of borrower's Decision.
        brw_D_idv_list = []
        gen = 0
        min_loan_term = self.Cal_Term_min_Repay_cap(P_min, interest_rate
                                , ann_income, saving_money, repay_capabil)
        #print("min_loan_term = "+str(min_loan_term))
        
        for cnt in range(population_size):
            D_idv_id = cnt+1
            
            if cnt == 0:
                Term = max_loan_term
                Price = P_min
            else:  
                Term = np.around(np.random.randint(min_loan_term,max_loan_term+1))
                
            
                ## Calculate the Price_max (upper bound of house price)
                Price_max_Repay_cap = self.Cal_Price_max_Repay_cap(Term, interest_rate
                                , ann_income, saving_money, repay_capabil)
            
                Price_limit = P_max
                if Price_max_Repay_cap < P_max:
                    Price_limit = Price_max_Repay_cap
                    
                if Price_limit < P_min:
                    Price_limit = P_min
            
                Price = np.around(np.random.randint(P_min,Price_limit+1))
            
            ## Set the LTV ratio value based on the selected house Price.
            if Price > saving_money:
                LTV_ratio = (Price - saving_money) / Price
            else:
                LTV_ratio = 0
            LTV_ratio = np.around(LTV_ratio, decimals=4)
            
            brw_D_idv = brw_Decision_idv(gen,D_idv_id,Price,Term,LTV_ratio,None)    
            brw_D_idv_list.append(brw_D_idv)
            
        return brw_D_idv_list
    
    
    def eval_brw_util(self, brw_D_idv_list
                      , eff_growth_rate, interest_rate
                      , ann_income, saving_money
                      , repay_capabil, max_loan_term):
        
        for brw_D_idv in brw_D_idv_list:  
            self.eval_brw_util_idv(brw_D_idv.Price, brw_D_idv.LTV_ratio
                               , eff_growth_rate, brw_D_idv.Term
                               , interest_rate)
        
        return brw_D_idv_list
    
    def eval_brw_util_idv(self, brw_D_idv
                      , eff_growth_rate, interest_rate
                      , ann_income, saving_money
                      , repay_capabil, max_loan_term):
        
        brw_obj = Borrower(0, ann_income, saving_money,
                           repay_capabil, max_loan_term)
        
        utility_value = brw_obj.utility_cal(brw_D_idv.Price, brw_D_idv.LTV_ratio
                                , eff_growth_rate, brw_D_idv.Term
                                , interest_rate)
        utility_value = np.around(utility_value)
        brw_D_idv.utility = utility_value
        
        return brw_D_idv
    
    ##-----------------------------------------------------------------------##
    ##  Find the best individual from the List.
    ##-----------------------------------------------------------------------##    
    def best_D_idv(self, D_idv_list):
        best_idv = None
        rnd = 0  ## rnd is used to keep track the number of round.
        for D_idv in D_idv_list:
            if rnd == 0:
                best_idv = D_idv
            
            if D_idv.utility > best_idv.utility:
                best_idv = D_idv
            
            ## increase the number of round
            rnd = rnd + 1
            
        return best_idv

    def gen_best_D_idv(self, D_idv_list, gen_best_idv_df):
        ## Find the best individual from the List.
        best_idv = self.best_D_idv(D_idv_list)

        # Transform the Object to the DataFrame.
        df_new = df_from_obj(best_idv)
        # Append the new Record into the DataFrame.
        gen_best_idv_df = pd.concat([gen_best_idv_df, df_new], ignore_index=True)
        
        return gen_best_idv_df
    
    ## Function for the Differential Evolutionary algorithm (DE) ##    
    def borrower_DE(self, max_LTV_ratio, interest_rate
                       , min_house_price, eff_growth_rate
                       , ann_income, saving_money, repay_capabil
                       , max_loan_term):
        
        gen = 0   ## id of the current generation.
        
        ngenf = 12  ## ngenf is number of generation, the iteration round.
        npf = 20  ## npf is number of population, the population size.
        CR = 0.9  ## CR is Crossover Rate.
        Ft = 0.8  ## Scaling Factor.
        
        P_min = min_house_price 
        
        
        if max_LTV_ratio >= 1:
            
            max_LTV_ratio = 0.9999
            
        P_max_1 = saving_money / (1 - max_LTV_ratio)
        
        P_max_2 = self.Cal_Price_max_Repay_cap(max_loan_term, interest_rate
                                , ann_income, saving_money, repay_capabil)
        
        
        ## P_max is the upper bound of the possible house price.
        P_max = P_max_1
        if P_max_2 < P_max_1:
            P_max = P_max_2
          
        
        # Initiate the population.
        brw_D_idv_list = self.init_ppl_brw_D_idv(npf, P_min, P_max
                           , max_loan_term 
                           , interest_rate, saving_money 
                           , ann_income, repay_capabil )
        
        # Evaluate the utility of the initial population.
        brw_D_idv_list = self.eval_brw_util(brw_D_idv_list
                      , eff_growth_rate, interest_rate
                      , ann_income, saving_money
                      , repay_capabil, max_loan_term)
        
        ### Collect the best individual of each generation. ###
        ## Find the best individual from the List.
        best_idv = self.best_D_idv(brw_D_idv_list)
            
        # Transform the Object to the DataFrame.
        gen_best_idv_df = df_from_obj(best_idv)        
        ######
        
        # Iterate the next generation.
        for gen in range(1, ngenf+1):
            
            nxt_brw_D_idv_list = []
            for idx in range(npf):
                
                # target individual
                tgt_idv = brw_D_idv_list[idx]
                
                # set the current generation number
                tgt_idv.generation = gen
                
                # Randomly select three other individuals
                r_idx_lst = []
                for r_idx in range(npf):
                    if r_idx != idx:
                        r_idx_lst.append(r_idx)
                shuffle(r_idx_lst)  # shuffle the index.
                
                r1_idv = brw_D_idv_list[r_idx_lst[0]]
                r2_idv = brw_D_idv_list[r_idx_lst[1]]
                r3_idv = brw_D_idv_list[r_idx_lst[2]]
                
                new_idv = brw_Decision_idv(gen, tgt_idv.D_idv_id
                                               ,tgt_idv.Price, tgt_idv.Term
                                               ,tgt_idv.LTV_ratio, None)
                
                # randomly select at least one parameter.
                n_par = 2 # n_par is number of parameter of input vector.
                inpar_Rand = np.random.randint(1,n_par+1)
                
                for inpar in range(1,n_par+1):
                    ## mutate each parameter of the input vector.
                    
                    rnd_zo = np.random.rand()
                    if (rnd_zo < CR) or (inpar == inpar_Rand):
                        
                        if inpar == 1:
                            Term_ch_1 = Ft*(r1_idv.Term - r2_idv.Term)
                            Term_ch_2 = Ft*(r3_idv.Term - tgt_idv.Term)
                            
                            new_idv.Term = tgt_idv.Term + Term_ch_1 + Term_ch_2
                            
                            # For integer
                            new_idv.Term = np.around(new_idv.Term)
                            
                            min_loan_term = self.Cal_Term_min_Repay_cap(P_min, interest_rate
                                , ann_income, saving_money, repay_capabil)
                            

                            ## bounded by the limit of the value of parameter
                            if new_idv.Term < min_loan_term:  
                                new_idv.Term = min_loan_term  ## lower bound
                            elif new_idv.Term > max_loan_term:  
                                new_idv.Term = max_loan_term  ## upper bound
                            
                            
                            
                        elif inpar == 2:
                            Price_ch_1 = Ft*(r1_idv.Price - r2_idv.Price)  
                            Price_ch_2 = Ft*(r3_idv.Price - tgt_idv.Price)
                            
                            new_idv.Price = tgt_idv.Price + Price_ch_1 + Price_ch_2
                            
                            
                            
                            # For integer
                            new_idv.Price = np.around(new_idv.Price) 
                            
                            Price_max_Repay_cap = self.Cal_Price_max_Repay_cap(new_idv.Term
                                , interest_rate
                                , ann_income, saving_money, repay_capabil)
                            
                            Price_limit = P_max
                            if Price_max_Repay_cap < P_max:
                                Price_limit = Price_max_Repay_cap
                                
                            
            
                            ## bounded by the limit of the value of parameter
                            if new_idv.Price < P_min:
                                new_idv.Price = P_min  ## lower bound
                            elif new_idv.Price > Price_limit:  
                                new_idv.Price = Price_limit  ## upper bound
                        
                            
                            
                            ## Set the LTV ratio value based on the selected house Price.
                            if new_idv.Price > saving_money:
                                new_LTV_ratio = (new_idv.Price - saving_money) / new_idv.Price
                            else:
                                new_LTV_ratio = 0
                            new_idv.LTV_ratio = np.around(new_LTV_ratio, decimals=4)
                
                # Evaluate the Utility of the new individual.
                new_idv = self.eval_brw_util_idv(new_idv
                      , eff_growth_rate, interest_rate
                      , ann_income, saving_money
                      , repay_capabil, max_loan_term)
                
                if new_idv.utility >= tgt_idv.utility:
                    nxt_brw_D_idv_list.append(new_idv)
                else:
                    nxt_brw_D_idv_list.append(tgt_idv)
            
            ## The current generation become the parent of the next generation.
            brw_D_idv_list = nxt_brw_D_idv_list
                          
            ## Collect the best individual of this generation.
            gen_best_idv_df = self.gen_best_D_idv(brw_D_idv_list, gen_best_idv_df) 
          
        brw_DE_output = [brw_D_idv_list, gen_best_idv_df]
        return brw_DE_output
       
        
    def borrower_decision(self, max_LTV_ratio, interest_rate
                       , min_house_price, eff_growth_rate
                       , ann_income, saving_money, repay_capabil
                       , max_loan_term, brw_buy_D):
        best_price = 0
        best_term = 0
        best_LTV_ratio = 0
        
        if brw_buy_D == True:
            ## DE for Borrower ##
            brw_DE_output = self.borrower_DE(max_LTV_ratio, interest_rate
                       , min_house_price, eff_growth_rate
                       , ann_income, saving_money, repay_capabil
                       , max_loan_term)
            
            brw_D_idv_list = brw_DE_output[0]
            gen_best_idv_df = brw_DE_output[1]
            
            ## number of row.
            n_row = gen_best_idv_df.shape[0]
            
            best_price = gen_best_idv_df["Price"].loc[n_row-1]
            best_term = gen_best_idv_df["Term"].loc[n_row-1]
            best_LTV_ratio = gen_best_idv_df["LTV_ratio"].loc[n_row-1]
            
        else:
            brw_D_idv_list = None
            gen_best_idv_df = None
            best_price = -1
            best_term = -1
            best_LTV_ratio = -1
            
        return [brw_D_idv_list, gen_best_idv_df, best_price, best_term, best_LTV_ratio]      

    def Cal_brw_Price_Term(self, max_LTV_ratio, interest_rate
                       , min_house_price, eff_growth_rate
                       , ann_income, saving_money, repay_capabil
                       , max_loan_term, brw_buy_D):
        
        Price, Term, LTV_ratio = 0 , 0 , 0 
        if brw_buy_D == True:
            if max_LTV_ratio >= 1:
                max_LTV_ratio = 0.9999
                
            P_max_1 = saving_money / (1 - max_LTV_ratio)
            P_max_2 = self.Cal_Price_max_Repay_cap(max_loan_term, interest_rate
                                , ann_income, saving_money, repay_capabil)
            P_max = P_max_1
            if P_max_2 < P_max_1:
                P_max = P_max_2
                
            Price = P_max 
            Term = max_loan_term
            LTV_ratio = (Price - saving_money) / Price 
            
            Price = np.around(Price, decimals=0)
            Term = np.around(Term, decimals=0)
            LTV_ratio = np.around(LTV_ratio, decimals=4)

        else:
            Price = -1
            Term = -1
            LTV_ratio = -1   
        
        return Price, Term, LTV_ratio
      
    ##-----------------------------------------------------------------------##
    ##  BankSector DE algortithm
    ##-----------------------------------------------------------------------##    
    def init_ppl_bkst_D_idv(self, population_size
                            , ir_min, ir_max ):
        ## initiate the population of the individual of BankSector's Decision.
        bkst_D_idv_list = []
        gen = 0
        
        multipier = 100
        ir_mult_max = ir_max*multipier
        ir_mult_min = ir_min*multipier
        
        for cnt in range(population_size):
            D_idv_id = cnt+1
            
            if cnt == 0:
                Int_rate = 0.05
            else:
                Int_rate = np.random.randint(ir_mult_min,ir_mult_max+1)
                Int_rate = Int_rate / multipier
                Int_rate = np.around(Int_rate, decimals=2)
            
            bkst_D_idv = bkst_Decision_idv(gen,D_idv_id,Int_rate,None)
            bkst_D_idv_list.append(bkst_D_idv)
        return bkst_D_idv_list
    
    def eval_bkst_util_idv(self, bkst_D_idv, bkst, brw_lst 
                       , min_house_price, max_LTV_ratio):
        Int_rate = bkst_D_idv.Int_rate
        utility_value = bkst.Cal_bkst_util(brw_lst, Int_rate
                , min_house_price, max_LTV_ratio
                , self.Cal_brw_Price_Term)
        utility_value = np.around(utility_value)
        bkst_D_idv.utility = utility_value
        
        ## Collect data of BankSector
        self.clct.Collect_bkst_Int_rate_result(bkst)
        
        return bkst_D_idv 
    
    def eval_bkst_util(self, bkst_D_idv_list, bkst, brw_lst 
                       , min_house_price, max_LTV_ratio):
        for bkst_D_idv in bkst_D_idv_list:
            self.eval_bkst_util_idv(bkst_D_idv, bkst, brw_lst 
                       , min_house_price, max_LTV_ratio)
        
        return bkst_D_idv_list  
        
    
    def BankSector_DE(self, min_house_price, max_LTV_ratio
                      , bkst, brw_lst):
        
        gen = 0   ## id of the current generation.

        
        ngenf = self.bkst_ngenf  ## ngenf is number of generation, the iteration round.
        npf = self.bkst_npf  ## npf is number of population, the population size.
        CR = self.bkst_CR  ## CR is Crossover Rate.
        Ft = self.bkst_Ft  ## Scaling Factor.
        
        ir_max = self.ir_max  ## maximum interest rate - upper bound
        ir_min = self.ir_min  ## minimum interest rate - lower bound
        
        # Initiate the population.
        bkst_D_idv_list = self.init_ppl_bkst_D_idv(npf,ir_min,ir_max)
                            
        
        # Evaluate the utility of the initial population.
        bkst_D_idv_list = self.eval_bkst_util(bkst_D_idv_list, bkst, brw_lst 
                       , min_house_price, max_LTV_ratio)
        
        
        ### Collect the best individual of each generation. ###
        ## Find the best individual from the List.
        best_idv = self.best_D_idv(bkst_D_idv_list)
          
        # Transform the Object to the DataFrame.
        gen_best_idv_df = df_from_obj(best_idv)        
        ######
        
        # Iterate the next generation.
        for gen in range(1, ngenf+1):
            
            nxt_bkst_D_idv_list = []
            
            for idx in range(npf):
                # target individual
                tgt_idv = bkst_D_idv_list[idx]
                
                # set the current generation number
                tgt_idv.generation = gen
                
                # Randomly select three other individuals
                r_idx_lst = []
                for r_idx in range(npf):
                    if r_idx != idx:
                        r_idx_lst.append(r_idx)
                shuffle(r_idx_lst)  # shuffle the index.
                
                r1_idv = bkst_D_idv_list[r_idx_lst[0]]
                r2_idv = bkst_D_idv_list[r_idx_lst[1]]
                r3_idv = bkst_D_idv_list[r_idx_lst[2]]
                
                new_idv = bkst_Decision_idv(gen,tgt_idv.D_idv_id
                                            ,tgt_idv.Int_rate,None)
                
                # randomly select at least one parameter.
                n_par = 1 # n_par is number of parameter of input vector.
                
                for inpar in range(1,n_par+1):
                    ## mutate each parameter of the input vector.
                    rnd_zo = np.random.rand()
                    
                    if (rnd_zo < CR):
                        
                        if inpar == 1:
                            Int_rate_ch_1 = Ft*(r1_idv.Int_rate - r2_idv.Int_rate)
                            Int_rate_ch_2 = Ft*(r3_idv.Int_rate - tgt_idv.Int_rate)
                            
                            new_idv.Int_rate = tgt_idv.Int_rate + Int_rate_ch_1 + Int_rate_ch_2
                            
                            # For integer
                            new_idv.Int_rate = np.around(new_idv.Int_rate, decimals=2)

                            ## bounded by the limit of the value of parameter
                            if new_idv.Int_rate < ir_min:  
                                new_idv.Int_rate = ir_min  ## lower bound
                            elif new_idv.Int_rate > ir_max:  
                                new_idv.Int_rate = ir_max  ## upper bound
                            
                # Evaluate the Utility of the new individual.
                new_idv = self.eval_bkst_util_idv(new_idv, bkst, brw_lst 
                       , min_house_price, max_LTV_ratio)
                
                if new_idv.utility >= tgt_idv.utility:
                    nxt_bkst_D_idv_list.append(new_idv)
                else:
                    nxt_bkst_D_idv_list.append(tgt_idv)
            
            ## The current generation become the parent of the next generation.
            bkst_D_idv_list = nxt_bkst_D_idv_list
                          
            ## Collect the best individual of this generation.
            gen_best_idv_df = self.gen_best_D_idv(bkst_D_idv_list, gen_best_idv_df) 
            
         ## number of row.
        n_row = gen_best_idv_df.shape[0]
            
        ## get the best Int_rate from the DataFrame.
        best_Int_rate = gen_best_idv_df["Int_rate"].loc[n_row-1]

        bkst_DE_output = [best_Int_rate, bkst_D_idv_list, gen_best_idv_df]
        return bkst_DE_output

    ##-----------------------------------------------------------------------##
    ##  Regulator DE algortithm
    ##-----------------------------------------------------------------------##    
    def init_ppl_regt_D_idv(self, population_size
                            , LTV_min, LTV_max ):
        ## initiate the population of the individual of BankSector's Decision.
        regt_D_idv_list = []
        gen = 0
        
        multipier = 100
        LTV_mult_max = LTV_max*multipier
        LTV_mult_min = LTV_min*multipier
        
        for cnt in range(population_size):
            D_idv_id = cnt+1
            
            if cnt == 0:
                Reg_LTV = 0.8
            else:
                Reg_LTV = np.random.randint(LTV_mult_min,LTV_mult_max+1)
                Reg_LTV = Reg_LTV / multipier
                Reg_LTV = np.around(Reg_LTV, decimals=2)
            
            print("Reg_LTV = "+ str(Reg_LTV))
            regt_D_idv = regt_Decision_idv(gen,D_idv_id,Reg_LTV,None)
            regt_D_idv_list.append(regt_D_idv)
        return regt_D_idv_list
    
    def eval_regt_util_idv(self, regt_D_idv, regt, bkst, brw_lst 
                        , min_house_price):
        Reg_LTV = regt_D_idv.Reg_LTV
        utility_value = regt.Cal_regt_util(self, min_house_price
                      , Reg_LTV, bkst, brw_lst)
        utility_value = np.around(utility_value, decimals=2)
        regt_D_idv.utility = utility_value
        
        ## Collect data of Regulator
        self.clct.Collect_regt_LTV_result(regt)
        
        return regt_D_idv
    
    def eval_regt_util(self, regt_D_idv_list, regt, bkst, brw_lst 
                       , min_house_price):
        for regt_D_idv in regt_D_idv_list:
            self.eval_regt_util_idv(regt_D_idv, regt, bkst, brw_lst 
                       , min_house_price)
        
        return regt_D_idv_list  
        
    
    def Regulator_DE(self, min_house_price, regt
                      , bkst, brw_lst):
        
        gen = 0   ## id of the current generation.
        
        print("regt gen = "+str(gen))
        print("timestamp : "+str(dt.datetime.now()))
        
        ngenf = 3  ## ngenf is number of generation, the iteration round.
        npf = 10  ## npf is number of population, the population size.
        CR = 0.9  ## CR is Crossover Rate.
        Ft = 0.8  ## Scaling Factor.
        
        LTV_max = self.LTV_max  ##0.99  ## maximum LTV ratio - upper bound
        LTV_min = self.LTV_min  ## minimum LTV ratio - lower bound
        
        # Initiate the population.
        regt_D_idv_list = self.init_ppl_regt_D_idv(npf,LTV_min,LTV_max)
                            
        
        # Evaluate the utility of the initial population.
        regt_D_idv_list = self.eval_regt_util(regt_D_idv_list
                    , regt, bkst, brw_lst , min_house_price)
        
        ### Collect the best individual of each generation. ###
        ## Find the best individual from the List.
        best_idv = self.best_D_idv(regt_D_idv_list)
          
        # Transform the Object to the DataFrame.
        gen_best_idv_df = df_from_obj(best_idv)        
        ######
        
        # Iterate the next generation.
        for gen in range(1, ngenf+1):
            print("regt gen = "+str(gen))
            print("timestamp : "+str(dt.datetime.now()))
            
            nxt_regt_D_idv_list = []
            
            for idx in range(npf):
                #print("individual idx: "+str(idx))
                # target individual
                tgt_idv = regt_D_idv_list[idx]
                
                # set the current generation number
                tgt_idv.generation = gen
                
                # Randomly select three other individuals
                r_idx_lst = []
                for r_idx in range(npf):
                    if r_idx != idx:
                        r_idx_lst.append(r_idx)
                shuffle(r_idx_lst)  # shuffle the index.
                
                r1_idv = regt_D_idv_list[r_idx_lst[0]]
                r2_idv = regt_D_idv_list[r_idx_lst[1]]
                r3_idv = regt_D_idv_list[r_idx_lst[2]]
                
                new_idv = regt_Decision_idv(gen,tgt_idv.D_idv_id
                                            ,tgt_idv.Reg_LTV,None)
                
                # randomly select at least one parameter.
                n_par = 1 # n_par is number of parameter of input vector.
                inpar_Rand = np.random.randint(1,n_par+1)
                
                for inpar in range(1,n_par+1):
                    ## mutate each parameter of the input vector.
                    #print("Input parameter : "+str(inpar))
                    rnd_zo = np.random.rand()
                    if (rnd_zo < CR) or (inpar == inpar_Rand):
                        
                        if inpar == 1:
                            Reg_LTV_ch_1 = Ft*(r1_idv.Reg_LTV - r2_idv.Reg_LTV)
                            Reg_LTV_ch_2 = Ft*(r3_idv.Reg_LTV - tgt_idv.Reg_LTV)
                            
                            new_idv.Reg_LTV = tgt_idv.Reg_LTV + Reg_LTV_ch_1 + Reg_LTV_ch_2

                            # For integer
                            new_idv.Reg_LTV = np.around(new_idv.Reg_LTV, decimals=2)

                            ## bounded by the limit of the value of parameter
                            if new_idv.Reg_LTV < LTV_min:  
                                new_idv.Reg_LTV = LTV_min  ## lower bound
                            elif new_idv.Reg_LTV > LTV_max:  
                                new_idv.Reg_LTV = LTV_max  ## upper bound
                            
                # Evaluate the Utility of the new individual.
                new_idv = self.eval_regt_util_idv(new_idv, regt, bkst, brw_lst 
                       , min_house_price)
                
                if new_idv.utility >= tgt_idv.utility:
                    nxt_regt_D_idv_list.append(new_idv)
                else:
                    nxt_regt_D_idv_list.append(tgt_idv)
            
            ## The current generation become the parent of the next generation.
            regt_D_idv_list = nxt_regt_D_idv_list
                          
            ## Collect the best individual of this generation.
            gen_best_idv_df = self.gen_best_D_idv(regt_D_idv_list, gen_best_idv_df) 
          
            ## Show the best value of decision variable.
            print("gen best Reg_LTV : "+str(gen_best_idv_df["Reg_LTV"].loc[gen_best_idv_df.shape[0]-1]))
            
        ## number of row.
        n_row = gen_best_idv_df.shape[0]
            
        ## get the best Int_rate from the DataFrame.
        best_Reg_LTV = gen_best_idv_df["Reg_LTV"].loc[n_row-1]

        regt_DE_output = [best_Reg_LTV, regt_D_idv_list, gen_best_idv_df]
        return regt_DE_output
    
    ##-----------------------------------------------------------------------##
    ##  One-Round Optimization (OneOpt) for Regulator.
    ##-----------------------------------------------------------------------##    

    def init_OneOpt_regt_D_idv_list(self, population_size
                            , LTV_min, LTV_max ):
        ## initiate the population of the individual of BankSector's Decision.
        regt_D_idv_list = []
        gen = 0
        
        multipier = 100
        LTV_mult_max = LTV_max*multipier
        LTV_mult_min = LTV_min*multipier
        
        for cnt in range(population_size):
            D_idv_id = cnt+1
            
            if cnt < 20:
                Reg_LTV = np.around(cnt*0.05 , decimals=2)
            elif cnt == 20:
                Reg_LTV = 1.0  ## 21st D_idv
            else:
                Reg_LTV = np.random.randint(LTV_mult_min,LTV_mult_max+1)
                Reg_LTV = Reg_LTV / multipier
                Reg_LTV = np.around(Reg_LTV, decimals=2)
                
            if Reg_LTV < LTV_min:
                Reg_LTV = LTV_min
            elif Reg_LTV > LTV_max:
                Reg_LTV = LTV_max
            
            regt_D_idv = regt_Decision_idv(gen,D_idv_id,Reg_LTV,None)
            regt_D_idv_list.append(regt_D_idv)
        return regt_D_idv_list
         

    def Regulator_OneOpt(self, min_house_price, regt
                      , bkst, brw_lst):
        
        print("timestamp : "+str(dt.datetime.now()))
        
        npf = 21  ## npf is number of population, the population size.
        
        LTV_max = self.LTV_max  ##0.99  ## maximum LTV ratio - upper bound
        LTV_min = self.LTV_min  ## minimum LTV ratio - lower bound
        
        # Initiate the population.
        regt_D_idv_list = self.init_OneOpt_regt_D_idv_list(npf,LTV_min,LTV_max)
                            
        
        # Evaluate the utility of the initial population.
        regt_D_idv_list = self.eval_regt_util(regt_D_idv_list
                    , regt, bkst, brw_lst , min_house_price)
        
        ### Collect the best individual of each generation. ###
        ## Find the best individual from the List.
        best_idv = self.best_D_idv(regt_D_idv_list)
        
        print("best_idv.Reg_LTV = "+str(best_idv.Reg_LTV))
          
        # Transform the Object to the DataFrame.
        gen_best_idv_df = df_from_obj(best_idv)        
        ######    
    
        ## number of row.
        n_row = gen_best_idv_df.shape[0]
            
        ## get the best Int_rate from the DataFrame.
        best_Reg_LTV = gen_best_idv_df["Reg_LTV"].loc[n_row-1]

        regt_OneOpt_output = [best_Reg_LTV, regt_D_idv_list, gen_best_idv_df]
        return regt_OneOpt_output
    
# =============================================================================
# Class of Scenario to set the env and optmz.
# =============================================================================
class Scenario(object):
    def __init__(self, w_alpha, w_beta, LTI_limit
                 , n_dstb_sample, n_brw, n_bank
                 , median_ann_income, min_Price_PTI_ratio
                 , mean_saving_coef, mean_repay_capabil
                 , mean_max_loan_term, mean_eff_growth_rate
                 , bkst_ngenf, bkst_npf, bkst_CR, bkst_Ft
                 , ir_max, ir_min, LTV_max, LTV_min
                     ): 
        ## weights of the social welfare value.
        self.w_alpha = w_alpha
        self.w_beta = w_beta
        
        ## LTI threshold of vulnerable loans.
        self.LTI_limit = LTI_limit
    
        ## Environment pararmeters.
        self.n_dstb_sample = n_dstb_sample
        self.n_brw = n_brw
        self.n_bank = n_bank
        self.median_ann_income = median_ann_income ## 30000
        self.min_Price_PTI_ratio = min_Price_PTI_ratio ## 2.5 
        self.mean_saving_coef = mean_saving_coef ## 0.4 
        self.mean_repay_capabil = mean_repay_capabil ## 0.4 
        self.mean_max_loan_term = mean_max_loan_term ## 300
        self.mean_eff_growth_rate = mean_eff_growth_rate ## 0.05 

        ## bkst DE algorithm parameters
        self.bkst_ngenf = bkst_ngenf  ## ngenf is number of generation, the iteration round.
        self.bkst_npf = bkst_npf ## npf is number of population, the population size.
        self.bkst_CR = bkst_CR ## CR is Crossover Rate.
        self.bkst_Ft = bkst_Ft ## Scaling Factor.
        self.ir_max = ir_max  ## upper bound of Interest rate
        self.ir_min = ir_min  ## lower bound of Interest rate

        ## regt OneOpt parameters
        self.LTV_max = LTV_max  ## upper bound of regulatory LTV ratio
        self.LTV_min = LTV_min  ## lower bound of regulatory LTV ratio
        
        ##-------------------------------------------------------------------##
        
        ## Output data of the three-level optimization.
        self.regt_best_Reg_LTV = None
        self.bkst_best_Int_rate = None
        
        self.n_brw_Good = None
        self.n_brw_Bad = None
        
        self.welfare_Good = None
        self.welfare_Bad = None
        self.regt_util_val = None
        
        self.brw_util_Good = None
        self.brw_util_Bad = None
        self.bank_profit_Good = None
        self.bank_profit_Bad = None
        
        self.n_brw_NotBuy = None
        self.n_brw_NotBorrow = None
        
# =============================================================================
# Class of UseCase to collect many scenarios.    
# =============================================================================
class UseCase(object):
    def __init__(self): 
        ## Use Case 1 - maximum term of loan
        self.usec_1_low_bound = 120
        self.usec_1_upp_bound = 480
        self.usec_1_step_size = 120
        self.usec_1_n_step = 4
        
        ## Use Case 2 - saving coefficient
        self.usec_2_low_bound = 0.0
        self.usec_2_upp_bound = 3.0
        self.usec_2_step_size = 0.4
        self.usec_2_n_step = 4
        
        ## Use Case 3 - min house price PTI ratio
        self.usec_3_low_bound =  1.0
        self.usec_3_upp_bound = 5.0
        self.usec_3_step_size = 1.0
        self.usec_3_n_step = 3
        
        ## Use Case 4 - repayment capability
        self.usec_4_low_bound = 0.0
        self.usec_4_upp_bound = 1.0
        self.usec_4_step_size = 0.2
        self.usec_4_n_step = 4
        
        ## Use Case 5 - Alpha weight
        self.usec_5_low_bound = 0.2
        self.usec_5_upp_bound = 0.8
        self.usec_5_step_size = 0.3
        self.usec_5_n_step = 3
        
        ## Use Case 6 - LTI limit
        self.usec_6_low_bound = 2.5
        self.usec_6_upp_bound = 6.5
        self.usec_6_step_size = 1.0
        self.usec_6_n_step = 4
        
        
    def init_scenario_list(self, bl_scen, usecase_set):
        scen_lst = []
        ## use case aging society.
        if usecase_set == 1:
            Tmax_low_bound = self.usec_1_low_bound
            Tmax_upp_bound = self.usec_1_upp_bound 
            step_size = self.usec_1_step_size   
            n_step = self.usec_1_n_step 
            
            for kk in range(n_step):
                Tmax_select = Tmax_low_bound + (kk*step_size)
                
                if Tmax_select > Tmax_upp_bound:
                    break
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = bl_scen.w_alpha
                                   , w_beta = bl_scen.w_beta
                                   , LTI_limit = bl_scen.LTI_limit
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = bl_scen.min_Price_PTI_ratio
                         , mean_saving_coef = bl_scen.mean_saving_coef
                         , mean_repay_capabil = bl_scen.mean_repay_capabil
                         , mean_max_loan_term = Tmax_select
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)
        ##---------------------------------------------------------------------        
        elif usecase_set == 2:
            saving_coef_low_bound = self.usec_2_low_bound
            saving_coef_upp_bound = self.usec_2_upp_bound 
            step_size = self.usec_2_step_size
            n_step = self.usec_2_n_step  
            
            for kk in range(n_step):
                saving_coef_select = saving_coef_low_bound + (kk*step_size)
                
                if saving_coef_select > saving_coef_upp_bound:
                    break
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = bl_scen.w_alpha
                                   , w_beta = bl_scen.w_beta
                                   , LTI_limit = bl_scen.LTI_limit
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = bl_scen.min_Price_PTI_ratio
                         , mean_saving_coef = saving_coef_select
                         , mean_repay_capabil = bl_scen.mean_repay_capabil
                         , mean_max_loan_term = bl_scen.mean_max_loan_term
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)    
        ##---------------------------------------------------------------------        
        elif usecase_set == 3:
            min_Price_ratio_low_bound =  self.usec_3_low_bound
            min_Price_ratio_upp_bound = self.usec_3_upp_bound
            step_size = self.usec_3_step_size
            n_step = self.usec_3_n_step    
            
            for kk in range(n_step):
                min_Price_ratio_select = min_Price_ratio_low_bound + (kk*step_size)
                
                if min_Price_ratio_select > min_Price_ratio_upp_bound:
                    break
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = bl_scen.w_alpha
                                   , w_beta = bl_scen.w_beta
                                   , LTI_limit = bl_scen.LTI_limit
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = min_Price_ratio_select
                         , mean_saving_coef = bl_scen.mean_saving_coef
                         , mean_repay_capabil = bl_scen.mean_repay_capabil
                         , mean_max_loan_term = bl_scen.mean_max_loan_term
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)    
        ##---------------------------------------------------------------------
        elif usecase_set == 4:
            repay_cpb_low_bound = self.usec_4_low_bound
            repay_cpb_upp_bound = self.usec_4_upp_bound
            step_size = self.usec_4_step_size 
            n_step = self.usec_4_n_step   
            
            for kk in range(n_step):
                repay_cpb_select = repay_cpb_low_bound + (kk*step_size)
                
                if repay_cpb_select > repay_cpb_upp_bound:
                    break
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = bl_scen.w_alpha
                                   , w_beta = bl_scen.w_beta
                                   , LTI_limit = bl_scen.LTI_limit
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = bl_scen.min_Price_PTI_ratio
                         , mean_saving_coef = bl_scen.mean_saving_coef
                         , mean_repay_capabil = repay_cpb_select
                         , mean_max_loan_term = bl_scen.mean_max_loan_term
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)    
        ##---------------------------------------------------------------------
        elif usecase_set == 5:
            w_alpha_low_bound = self.usec_5_low_bound
            w_alpha_upp_bound = self.usec_5_upp_bound
            step_size = self.usec_5_step_size 
            n_step = self.usec_5_n_step     
            
            for kk in range(n_step):
                w_alpha_select = w_alpha_low_bound + (kk*step_size)
                
                if w_alpha_select > w_alpha_upp_bound:
                    break
                
                w_beta_select = 1 - w_alpha_select
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = w_alpha_select
                                   , w_beta = w_beta_select
                                   , LTI_limit = bl_scen.LTI_limit
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = bl_scen.min_Price_PTI_ratio
                         , mean_saving_coef = bl_scen.mean_saving_coef
                         , mean_repay_capabil = bl_scen.mean_repay_capabil
                         , mean_max_loan_term = bl_scen.mean_max_loan_term
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)    
        ##---------------------------------------------------------------------
        elif usecase_set == 6:
            LTI_limit_low_bound = self.usec_6_low_bound
            LTI_limit_upp_bound = self.usec_6_upp_bound 
            step_size = self.usec_6_step_size
            n_step = self.usec_6_n_step    
            
            for kk in range(n_step):
                LTI_limit_select = LTI_limit_low_bound + (kk*step_size)
                
                if LTI_limit_select > LTI_limit_upp_bound:
                    break
                
                ## initiate new scenario.
                new_scen = Scenario(w_alpha = bl_scen.w_alpha
                                   , w_beta = bl_scen.w_beta
                                   , LTI_limit = LTI_limit_select
                         , n_dstb_sample = bl_scen.n_dstb_sample
                         , n_brw = bl_scen.n_brw, n_bank = bl_scen.n_bank
                         , median_ann_income = bl_scen.median_ann_income
                         , min_Price_PTI_ratio = bl_scen.min_Price_PTI_ratio
                         , mean_saving_coef = bl_scen.mean_saving_coef
                         , mean_repay_capabil = bl_scen.mean_repay_capabil
                         , mean_max_loan_term = bl_scen.mean_max_loan_term
                         , mean_eff_growth_rate = bl_scen.mean_eff_growth_rate
                         , bkst_ngenf = bl_scen.bkst_ngenf
                         , bkst_npf = bl_scen.bkst_npf
                         , bkst_CR = bl_scen.bkst_CR, bkst_Ft = bl_scen.bkst_Ft
                         , ir_max = bl_scen.ir_max, ir_min = bl_scen.ir_min
                         , LTV_max = bl_scen.LTV_max, LTV_min = bl_scen.LTV_min )
                ## add new scenario into the list.
                scen_lst.append(new_scen)    
        ##---------------------------------------------------------------------
        return scen_lst
    
# =============================================================================
# Class of Simulator to simulate the scenario.    
# =============================================================================
class Simulator(object):
    def __init__(self): 
        ## variables to keep the object from function set_scenario().
        self.env = None
        self.optmz = None
        
        ## Enable Simulation.
        self.enb_sim = True
        
        ## Enable showing text.
        self.show_is_enable = False
        
        ## Number of round for multiple simulation.
        self.stat_round = 100
        
        ## Enable to collect data using Data Pack.
        self.enb_datpck = False
        
    # -------------------------------------------------------------------------
    def set_regt(self, scen):
        self.env.regt = Regulator(regr_id = "CB"
                                  , w_alpha = scen.w_alpha
                                  , w_beta = scen.w_beta
                                  , LTI_limit = scen.LTI_limit)
        
    def set_env(self, scen):
        self.env.n_dstb_sample = scen.n_dstb_sample
        self.env.n_brw =  scen.n_brw
        self.env.n_bank = scen.n_bank
        self.env.median_ann_income = scen.median_ann_income
        self.env.min_Price_PTI_ratio = scen.min_Price_PTI_ratio 
        self.env.mean_saving_coef = scen.mean_saving_coef
        self.env.mean_repay_capabil =  scen.mean_repay_capabil
        self.env.mean_max_loan_term = scen.mean_max_loan_term
        self.env.mean_eff_growth_rate = scen.mean_eff_growth_rate
        
    def set_optmz(self, scen):
        self.optmz.bkst_ngenf = scen.bkst_ngenf
        self.optmz.bkst_npf = scen.bkst_npf
        self.optmz.bkst_CR = scen.bkst_CR
        self.optmz.bkst_Ft = scen.bkst_Ft
        self.optmz.ir_max = scen.ir_max
        self.optmz.ir_min = scen.ir_min
        self.optmz.LTV_max = scen.LTV_max
        self.optmz.LTV_min = scen.LTV_min
        
    def set_scenario(self, scen):
        ## Reset env and optmz.
        self.env = Environment()
        self.optmz = Optimizer(DataCollector())
        
        ## Set scenario.
        self.set_regt(scen)
        self.set_env(scen)
        self.set_optmz(scen)
        
    # -------------------------------------------------------------------------   
    def sim_setup(self):
        env = self.env
        
        ## Create the distribution of income.
        env.set_ann_income_dstb(env.n_dstb_sample)

        ## Create the list of the Borrower object
        env.init_borrowerList(env.n_brw)

        ## Set the minimum house price from the median of income distribution. 
        env.set_min_house_price()
        
        ## Initiate the objects from the classes - BankSector.
        env.bkst = BankSector(env.n_bank)
        
        ## Get the data of each Borrower into a DataFrame.
        env.regt.df_brw_lst = df_from_obj_list(env.brw_lst)
    
    def get_env_dstb_data(self, datpck):
        if self.enb_datpck == True:
            env = self.env
            
            ## get the distribution data in Environment object.
            datpck.env_income_dstb = env.income_dstb
            datpck.env_sample_income_dstb = env.sample_income_dstb
            datpck.env_saving_coef_arr = env.saving_coef_arr
            datpck.env_repay_capabil_arr = env.repay_capabil_arr
            datpck.env_max_loan_term_arr = env.max_loan_term_arr
            datpck.env_eff_growth_rate_arr = env.eff_growth_rate_arr
        return datpck
        
    def sim_oneopt(self):
        env = self.env
        optmz = self.optmz
        # =====================================================================
        # One-Round Optimization for Regulator
        # =====================================================================
        print("-------------------------------------------------------")
        print("Regulator OneOpt start : "+str(dt.datetime.now()))
        # ---------------------------------------------------------------------
        optmz.clct.is_regt_enable = True

        regt_OneOpt_output = optmz.Regulator_OneOpt(env.min_house_price
                                                , env.regt
                                                , env.bkst, env.brw_lst)

        optmz.clct.is_regt_enable = False
        # ---------------------------------------------------------------------
        print("Regulator OneOpt finish : "+str(dt.datetime.now()))
        print("-------------------------------------------------------")
        
        return regt_OneOpt_output
    
    # ------------------------------------------------------------------------- 
    def get_optm_result(self, regt_OneOpt_output, scen, datpck):
        optmz = self.optmz
        env = self.env
        
        # --------------------------------------------------------------
        # Get the optimal results.
        # --------------------------------------------------------------
        scen.regt_best_Reg_LTV = regt_OneOpt_output[0]
        
        # ----------------------------------------------------
        # Simulate the optimal decision one more time.
        # ----------------------------------------------------
        ## Enable the DataCollector in the Optimizer.
        optmz.clct.is_bkst_enable = True

        ## Calculate utility value of the optimal LTV ratio.
        scen.regt_util_val = env.regt.Cal_regt_util(optmz, env.min_house_price
                                       , scen.regt_best_Reg_LTV
                                       , env.bkst, env.brw_lst)

        ## Disable the DataCollector in the Optimizer.
        optmz.clct.is_bkst_enable = False
        
        # ----------------------------------------------------
        # Get the data about banks and borrowers.
        # ----------------------------------------------------
        scen.n_brw_Good = env.regt.n_brw_Good
        scen.n_brw_Bad = env.regt.n_brw_Bad
        # ----------------------------------------
        scen.welfare_Good = env.regt.welfare_Good
        scen.welfare_Bad = env.regt.welfare_Bad
        scen.brw_util_Good = env.regt.brw_util_Good
        scen.brw_util_Bad = env.regt.brw_util_Bad
        scen.bank_profit_Good = env.regt.bank_profit_Good
        scen.bank_profit_Bad = env.regt.bank_profit_Bad
        # ----------------------------------------
        bkst_DE_output = env.regt.bkst_DE_output
        scen.bkst_best_Int_rate = bkst_DE_output[0]
        # ----------------------------------------
        scen.n_brw_NotBuy = env.regt.n_brw_NotBuy
        scen.n_brw_NotBorrow = env.regt.n_brw_NotBorrow
        # ----------------------------------------------------
        
        if self.enb_datpck == True:
            # -----------------------------------------------------------------
            # Build a DataFrame to show the best LTV ratio.
            # -----------------------------------------------------------------
            regt_D_idv_list = regt_OneOpt_output[1]
            datpck.df_regt_D_idv_list = df_from_obj_list(regt_D_idv_list)
            
            # -----------------------------------------------------------------
            # Get data about optimization over generation of BankSector.
            # -----------------------------------------------------------------
            datpck.bkst_gen_best_idv_df = bkst_DE_output[2]
            
            # =================================================================
            # Get the DataFrame in the DataCollector object.
            # =================================================================
            datpck.df_bkst_Int_rate_res = optmz.clct.bkst_Int_rate_res
            datpck.df_regt_LTV_res = optmz.clct.regt_LTV_res
            
            # -----------------------------------------------------------------
            
        return scen, datpck
        # ----------------------------------------------------

    # -------------------------------------------------------------------------    
    def show_env_var(self):
        env = self.env
        # =====================================================================
        # Show the value of this scenario.   
        # =====================================================================
        print("-------------------------------------------------------")
        print("env.n_dstb_sample = " + str(env.n_dstb_sample) ) 
        print("env.n_brw = " + str(env.n_brw) ) 
        print("env.n_bank = " + str(env.n_bank) ) 
        print("-------------------------------------------------------")

        print("Median Annual Income = " + str(np.median(env.income_dstb)) ) 
        print("Median Sampled Annual Income = " + str(np.median(env.sample_income_dstb)) ) 
        print("env.median_ann_income = " + str(env.median_ann_income) ) 

        print("env.min_Price_PTI_ratio = " + str(env.min_Price_PTI_ratio) ) 
        print("env.min_house_price = " + str(env.min_house_price) ) 
        print("-------------------------------------------------------")

        print("env.mean_saving_coef = " + str(env.mean_saving_coef) ) 
        print("env.mean_repay_capabil = " + str(env.mean_repay_capabil) ) 
        print("env.mean_max_loan_term = " + str(env.mean_max_loan_term) ) 
        print("env.mean_eff_growth_rate = " + str(env.mean_eff_growth_rate) ) 
        print("-------------------------------------------------------")

        print("median(env.saving_coef_arr) = " + str(np.around(
                np.median(env.saving_coef_arr), decimals=4)) ) 
        print("median(env.repay_capabil_arr) = " + str(np.around(
                np.median(env.repay_capabil_arr), decimals=4)) ) 
        print("median(env.max_loan_term_arr) = " + str(np.around(
                np.median(env.max_loan_term_arr), decimals=0)) ) 
        print("median(env.eff_growth_rate_arr) = " + str(np.around(
                np.median(env.eff_growth_rate_arr), decimals=4)) ) 
        print("-------------------------------------------------------")

    def show_scen_result_var(self, scen):
        env = self.env
        # -----------------------------------------------------------------------------
        # Show the Regulator optimal results.
        # -----------------------------------------------------------------------------
        print("-------------------------------------------------------")
        ## the optimal decision.
        print("regt_best_Reg_LTV = "+str(scen.regt_best_Reg_LTV))
        
        ## the utility of that optimal decision.
        print("regt_util_val = "+str(scen.regt_util_val))
        print("env.regt.util_val = "+str(env.regt.util_val))
        
        # -----------------------------------------------------------------------------
        ## Show the data about banks and borrowers.
        print("n_brw_Good = "+str(scen.n_brw_Good))
        print("n_brw_Bad = "+str(scen.n_brw_Bad))
        print("n_brw_NotBuy = "+str(scen.n_brw_NotBuy))
        print("n_brw_NotBorrow = "+str(scen.n_brw_NotBorrow))
        
        print("bkst_best_Int_rate = "+str(scen.bkst_best_Int_rate))
        
        # -----------------------------------------------------------------------------
        print("-------------------------------------------------------")
        
    ## ------------------------------------------------------------ ##
    def create_new_scen(self, scen):
        new_scen = Scenario(w_alpha = scen.w_alpha
                                   , w_beta = scen.w_beta
                                   , LTI_limit = scen.LTI_limit
                         , n_dstb_sample = scen.n_dstb_sample
                         , n_brw = scen.n_brw, n_bank = scen.n_bank
                         , median_ann_income = scen.median_ann_income
                         , min_Price_PTI_ratio = scen.min_Price_PTI_ratio
                         , mean_saving_coef = scen.mean_saving_coef
                         , mean_repay_capabil = scen.mean_repay_capabil
                         , mean_max_loan_term = scen.mean_max_loan_term
                         , mean_eff_growth_rate = scen.mean_eff_growth_rate
                         , bkst_ngenf = scen.bkst_ngenf
                         , bkst_npf = scen.bkst_npf
                         , bkst_CR = scen.bkst_CR, bkst_Ft = scen.bkst_Ft
                         , ir_max = scen.ir_max, ir_min = scen.ir_min
                         , LTV_max = scen.LTV_max, LTV_min = scen.LTV_min )
        return new_scen
    
    ## ============================================================ ##
    def simulate_a_scene(self, input_scen):
        ## initiate DataPack object.
        datpck = DataPack()
        
        ## create a new scen object.
        scen = self.create_new_scen(input_scen)
        
        if self.enb_sim == True:
            ## ----------------------------------------------------------------
            ## setup env variables.
            self.sim_setup()
            
            ## get data of distributions in Environment.
            datpck = self.get_env_dstb_data(datpck)
            
            ## show the value of this scenario. 
            if self.show_is_enable == True:
                self.show_env_var()
            
            ## One-Round Optimization of Regulator's decision.
            regt_OneOpt_output = self.sim_oneopt()
            
            ## get results of the optimal decision.
            scen, datpck = self.get_optm_result(regt_OneOpt_output, scen, datpck)
            
            ## show the results of this scenario.
            if self.show_is_enable == True:
                self.show_scen_result_var(scen)
            ## ----------------------------------------------------------------
            
        return scen, datpck
    
    # -------------------------------------------------------------------------
    def simulate_many_scenes(self, scen_lst):
        datpck_lst = []
        
        ## for-loop to set and simulate eacn scenario.
        out_scen_lst = []
        for ii in range(len(scen_lst)):
            ## set scenario and simulate.
            scen = scen_lst[ii]
            print("scen ii : "+str(ii))
            
            self.set_scenario(scen)
            scen, datpck = self.simulate_a_scene(scen)

            ## add the scenario to the list.
            out_scen_lst.append(scen)
            
            ## get the DataPack into a list.
            datpck_lst.append(datpck)
            
        ## get the result of eacn scenario.
        df_out_scen_lst = df_from_obj_list(out_scen_lst)
        
        return df_out_scen_lst, datpck_lst
    
    # -------------------------------------------------------------------------
    def multiple_simulate(self, scen_lst):
        stat_rnd = self.stat_round  ## Default 100 rounds
        datpck_lst = []
        
        ## for-loop to set and simulate eacn scenario.
        out_scen_lst = []
        for ii in range(len(scen_lst)):
            ## set scenario and simulate multiple round.
            scen = scen_lst[ii]
            print("scen ii : "+str(ii))
            
            for rnd in range(stat_rnd): 
                print("scen rnd : "+str(rnd))
                
                self.set_scenario(scen)
                scen, datpck = self.simulate_a_scene(scen)

                ## add the scenario to the list.
                out_scen_lst.append(scen)
            
                ## get the DataPack into a list.
                datpck_lst.append(datpck)
            
        ## get the result of eacn scenario.
        df_out_scen_lst = df_from_obj_list(out_scen_lst)
        
        return df_out_scen_lst, datpck_lst
        
# =============================================================================
# Class of Data Pack
# =============================================================================
## Dataset in this Data Pack will be visualised in graphs.
        
class DataPack(object):
    def __init__(self):
        ## For graph of distributions of borrower's attributes.
        self.env_income_dstb = None
        self.env_sample_income_dstb = None
        self.env_saving_coef_arr = None
        self.env_repay_capabil_arr = None
        self.env_max_loan_term_arr = None
        self.env_eff_growth_rate_arr = None
        
        ## For graph of BankSector improvement over generation.
        self.bkst_gen_best_idv_df = None
        ## For graph the best LTV ratio.     
        self.df_regt_D_idv_list = None
        ## For graph of the utility over different interest rate.
        self.df_bkst_Int_rate_res = None
        
        ## For graph of the n_brw_NotBuy over LTV ratio.
        self.df_regt_LTV_res = None

# =============================================================================
# Class of Visualiser
# =============================================================================
## Visualiser contains fuctions to plot graphs.
        
class Visualiser(object):
    def __init__(self):
        self.fig_number = 1
        self.pic_filepath = "../pic/"

    def plot_graphs(self, datpck):
        ## Plot graphs of distributions
        ## , BankSector results, and Regulator results.
        self.plot_env_dstb(datpck)
        self.plot_bkst_res(datpck)
        self.plot_regt_res(datpck)
     
    # -------------------------------------------------------------------------
    def plot_env_dstb(self, datpck):
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
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_income_dstb, bins=n_bins, alpha=0.8)
        plt.title("Annual Income Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.grid()
        #plt.savefig(self.pic_filepath+'income_dist.png', format='png')
        plt.show() 
        
        self.fig_number = self.fig_number + 1
        
        ## =============================================================================
        ### Plot the histogram of sample income distribution.
        ## =============================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_sample_income_dstb, bins=n_bins, alpha=0.8)
        plt.title("Sampled Annual Income Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.grid()
        #plt.savefig(self.pic_filepath+'sample_income_dist.png', format='png')
        plt.show() 
        
        self.fig_number = self.fig_number + 1
        
        # =============================================================================
        # Plot the distributions.
        # =============================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_saving_coef_arr , bins=n_bins, alpha=0.8)
        plt.title("Saving Coefficient Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.show() 
        self.fig_number = self.fig_number + 1
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_repay_capabil_arr , bins=n_bins, alpha=0.8)
        plt.title("Repayment Capability Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.show()
        self.fig_number = self.fig_number + 1 
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_max_loan_term_arr , bins=n_bins, alpha=0.8)
        plt.title("Max Loan Term Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.show() 
        self.fig_number = self.fig_number + 1
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.hist(env_eff_growth_rate_arr , bins=n_bins, alpha=0.8)
        plt.title("Effective Growth Rate Histogram", fontsize="20")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        plt.show() 
        self.fig_number = self.fig_number + 1
    
    # -------------------------------------------------------------------------     
    def plot_bkst_res(self, datpck): 
        ## For graph of BankSector improvement over generation.
        bkst_gen_best_idv_df =  datpck.bkst_gen_best_idv_df
        
        ## For graph of the utility over different interest rate.
        df_bkst_Int_rate_res =  datpck.df_bkst_Int_rate_res
    
        # =====================================================================
        # Plot graph of the improvement over generation.
        # =====================================================================   
        
        fig = plt.figure(self.fig_number)
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
        #plt.savefig(self.pic_filepath+'bkst_util_improve_gen.png', format='png')
        plt.show()
        
        self.fig_number = self.fig_number + 1
        # ---------------------------------------------------------------------
       
        # =====================================================================
        # Plot graph of the utility over different interest rate.
        # =====================================================================
        
        ## Sorting value in the DataFrame.
        df_bkst_Int_rate_res = df_bkst_Int_rate_res.sort_values(by=["bkst_Int_rate"])
        
        ## This is the response under the best regulatory LTV ratio.
        fig = plt.figure(self.fig_number)
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
        #plt.savefig(self.pic_filepath+'bkst_util_on_Int_rate.png', format='png')
        plt.show()
        
        self.fig_number = self.fig_number + 1 
        # ---------------------------------------------------------------------
        # Plot BankSector utility/profit, loan amount, and total repayment.
        # ---------------------------------------------------------------------
        
        fig = plt.figure(self.fig_number)
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
        plt.show()
        
        self.fig_number = self.fig_number + 1 
        
    # -------------------------------------------------------------------------   
    def plot_regt_res(self, datpck): 
        ## For graph of Regulator utility over different LTV ratios.
        df_regt_D_idv_list =  datpck.df_regt_D_idv_list
        
        ## For graph of the n_brw_NotBuy over LTV ratios.
        df_regt_LTV_res = datpck.df_regt_LTV_res
        # ---------------------------------------------------------------------
   
        # =============================================================================
        # Plot graph of the utility over regulatory LTV ratio.
        # =============================================================================
        
        fig = plt.figure(self.fig_number)
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
        #plt.savefig(self.pic_filepath+'regt_util_on_LTV_ratio.png', format='png')
        plt.show()
        
        self.fig_number = self.fig_number + 1    
        
        # -----------------------------------------------------------------------------
        # Plot graph of Good welfare over regulatory LTV ratios.
        # -----------------------------------------------------------------------------
        fig = plt.figure(self.fig_number)
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
        plt.show()
        
        self.fig_number = self.fig_number + 1 
        # -----------------------------------------------------------------------------
        
        # -----------------------------------------------------------------------------
        # Plot graph of n_brw_NotBuy over regulatory LTV ratios.
        # -----------------------------------------------------------------------------
        fig = plt.figure(self.fig_number)
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
        plt.show()
        
        self.fig_number = self.fig_number + 1 
    
    # -------------------------------------------------------------------------   
    def plot_usecase_1_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["mean_max_loan_term"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over maximum term of loan", fontsize="20")
        
        plt.xlabel("Mean of Maximum Term of Loan", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1    

    # -------------------------------------------------------------------------   
    def plot_usecase_2_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["mean_saving_coef"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over saving coefficient", fontsize="20")
        
        plt.xlabel("Mean of Saving Coefficient", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1     
        
    # -------------------------------------------------------------------------   
    def plot_usecase_3_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["min_Price_PTI_ratio"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over minimum house price coefficient"
                  , fontsize="20")
        
        plt.xlabel("Minimum House Price Coefficient", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1    
        
    # -------------------------------------------------------------------------   
    def plot_usecase_4_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["mean_repay_capabil"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over repayment capability", fontsize="20")
        
        plt.xlabel("Mean of Repayment Capability", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1       
               
    # -------------------------------------------------------------------------   
    def plot_usecase_5_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["w_alpha"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over Alpha weight", fontsize="20")
        
        plt.xlabel("Alpha weight", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1       
                      
    # -------------------------------------------------------------------------   
    def plot_usecase_6_res(self, df_out_scen_lst): 
        # =====================================================================
        # Plot graph of the optimal LTV ratio over variable value.
        # =====================================================================
        
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        plt.plot(df_out_scen_lst["LTI_limit"]
                , df_out_scen_lst["regt_best_Reg_LTV"]
                , 'bx', linewidth=2, markersize=8)
        plt.title("Optimal LTV ratio over LTI limit", fontsize="20")
        
        plt.xlabel("LTI limit", fontsize="20")
        plt.ylabel("Optimal LTV ratio", fontsize="20")

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.show()
        
        self.fig_number = self.fig_number + 1   

    # -------------------------------------------------------------------------    
    # Boxplot visualisation.    
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    def boxplot_multisim_res(self, df_x, df_y
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
        fig = plt.figure(self.fig_number)
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
        plt.savefig(self.pic_filepath + plt_title + '.png', format='png')
        plt.show()
        
        self.fig_number = self.fig_number + 1    
    
    # ------------------------------------------------------------------------- 
    def three_boxplots(self, df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst):
        
        ## result of the optimal regulatory LTV ratio 
        df_y = df_out_scen_lst["regt_best_Reg_LTV"]
        plt_title = "Optimal LTV ratio " + title_suffix
        
        plt_ylabel = "Optimal LTV ratio"
        bp_ylegend = "median value"
        self.boxplot_multisim_res(df_x, df_y
                              , plt_title, plt_xlabel, plt_ylabel
                              , bp_ylegend)
        
        ## result of the optimal interest rate
        df_y = df_out_scen_lst["bkst_best_Int_rate"]
        plt_title = "Optimal interest rate " + title_suffix
        plt_ylabel = "Optimal Interest Rate"
        bp_ylegend = "median value" 
        self.boxplot_multisim_res(df_x, df_y
                              , plt_title, plt_xlabel, plt_ylabel
                              , bp_ylegend)
        
        ## result of the people that cannot buy a house
        df_y = 100*(df_out_scen_lst["n_brw_NotBuy"]/df_out_scen_lst["n_brw"])
        plt_title = "People who did not buy a house " + title_suffix
        plt_ylabel = "% of people"
        bp_ylegend = "median value" 
        self.boxplot_multisim_res(df_x, df_y
                              , plt_title, plt_xlabel, plt_ylabel
                              , bp_ylegend)
        
    # ------------------------------------------------------------------------- 
    def boxplot_usecase_1_res(self, df_out_scen_lst):
        df_x = df_out_scen_lst["mean_max_loan_term"]
        plt_xlabel = "Mean of Maximum Term of Loan"
        title_suffix = "over maximum term of loan"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)
       
    # ------------------------------------------------------------------------- 
    def boxplot_usecase_2_res(self, df_out_scen_lst):
        df_x = df_out_scen_lst["mean_saving_coef"]
        plt_xlabel = "Mean of Saving Coefficient"
        title_suffix = "over saving coefficient"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)
                
    # ------------------------------------------------------------------------- 
    def boxplot_usecase_3_res(self, df_out_scen_lst): 
        df_x = df_out_scen_lst["min_Price_PTI_ratio"]
        plt_xlabel = "Minimum House Price Coefficient"
        title_suffix = "over minimum house price coefficient"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)
        
    def one_boxplot_usecase_3_res(self, df_out_scen_lst): 
        ## result of the optimal regulatory LTV ratio 
        df_x = df_out_scen_lst["min_Price_PTI_ratio"]
        df_y = df_out_scen_lst["regt_best_Reg_LTV"]
        plt_title = "Optimal LTV ratio over minimum house price coefficient"
        plt_xlabel = "Minimum House Price Coefficient"
        plt_ylabel = "Optimal LTV ratio"
        bp_ylegend = "median value"
        self.boxplot_multisim_res(df_x, df_y
                              , plt_title, plt_xlabel, plt_ylabel
                              , bp_ylegend)

    # -------------------------------------------------------------------------
    def boxplot_usecase_4_res(self, df_out_scen_lst): 
        df_x = df_out_scen_lst["mean_repay_capabil"]
        plt_xlabel = "Mean of Repayment Capability"
        title_suffix = "over repayment capability"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)    
        
    # -------------------------------------------------------------------------
    def boxplot_usecase_5_res(self, df_out_scen_lst): 
        df_x = df_out_scen_lst["w_alpha"]
        plt_xlabel = "Alpha weight"
        title_suffix = "over Alpha weight"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)
        
    # -------------------------------------------------------------------------
    def boxplot_usecase_6_res(self, df_out_scen_lst): 
        df_x = df_out_scen_lst["LTI_limit"]
        plt_xlabel = "LTI limit"
        title_suffix = "over LTI limit"
        
        self.three_boxplots(df_x, plt_xlabel
                       , title_suffix, df_out_scen_lst)
        
    # -------------------------------------------------------------------------
    def boxplot_baseline_res(self, df_out_scen_lst): 
        title_suffix = "in the baseline scenario"
        
        ## result of the optimal regulatory LTV ratio 
        df_y = df_out_scen_lst["regt_best_Reg_LTV"]
        plt_title = "Optimal LTV ratio " + title_suffix
        plt_ylabel = "Optimal LTV ratio"
        bp_ylegend = "Median Optimal LTV ratio" 
        self.boxplot_onebox_res(df_y, plt_title, plt_ylabel, bp_ylegend)
        
        ## result of the optimal interest rate
        df_y = df_out_scen_lst["bkst_best_Int_rate"]
        plt_title = "Optimal interest rate " + title_suffix
        plt_ylabel = "Optimal Interest Rate"
        bp_ylegend = "median value" 
        self.boxplot_onebox_res(df_y, plt_title, plt_ylabel, bp_ylegend)
        
        ## result of the people that cannot buy a house
        df_y = 100*(df_out_scen_lst["n_brw_NotBuy"]/df_out_scen_lst["n_brw"])
        plt_title = "People who did not buy a house " + title_suffix
        plt_ylabel = "% of people"
        bp_ylegend = "median value" 
        self.boxplot_onebox_res(df_y, plt_title, plt_ylabel, bp_ylegend)
        
    # -------------------------------------------------------------------------
    def boxplot_onebox_res(self, df_y, plt_title, plt_ylabel, bp_ylegend): 
        ## Prep data for boxplot function.        
        n_rnd = len(df_y)
        n_box = 1
            
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
    
        # --------------------------------------------------------------------- 
        # Plot graph
        # ---------------------------------------------------------------------
        fig = plt.figure(self.fig_number)
        fig, ax = plt.subplots(figsize=(10, 5))

        ## line graph
        plt.plot([1], med_lst, '--o'
                 , label=bp_ylegend)
        
        ## boxplot
        bp = ax.boxplot(x=data_lst
                    , notch=0, vert=1, whis=1.5)
        
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='black', marker='.')
        plt.setp(bp['medians'], color='red')
        
        plt.title(plt_title, fontsize="20")
        
        plt.xlabel("", fontsize="20")
        plt.ylabel(plt_ylabel, fontsize="20")

        plt.xticks([1], [''] , fontsize=16)
        plt.yticks(fontsize=16)     
        plt.grid()
        ax.set_axisbelow(True)
        plt.legend()
        plt.savefig(self.pic_filepath + plt_title + '.png', format='png')
        plt.show()
        
        self.fig_number = self.fig_number + 1    
            
    # -------------------------------------------------------------------------
    
        
# -----------------------------------------------------------------------------
# Function to save and load object.
# -----------------------------------------------------------------------------
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output) #, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj   

# -----------------------------------------------------------------------------
# Function to save list of text to csv and to load from csv .
# -----------------------------------------------------------------------------
def save_txt_list(lst, filename):
    with open(filename, 'w') as filehandle:  
        for listitem in lst:
            filehandle.write('%s\n' % listitem) 

def load_txt_list(path):
    # define an empty list
    lst = []
    # open file and read the content in a list
    with open(path, 'r') as filehandle:  
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
    
            # add item to the list
            lst.append(currentPlace) 
    return lst
    
# -----------------------------------------------------------------------------
      
## ****************************************************************************
## Test the Classes and their Functions.
## ****************************************************************************
get_dt_start_time = dt.datetime.now()
print("-------------------------------------------------------")
print("start time : "+str(get_dt_start_time))
print("-------------------------------------------------------")
# =============================================================================
# A Stackelberg Mortgage Game Simulation.
# =============================================================================
enb_main = True

## ---------------------------------------------------- ##
## For Baseline scenario simulation.
enb_a_sim_bsln = False     ## one round simulation.
enb_multi_sim_bsln = True   ## multiple simulation.

## save and loan DataPack of one round simulation.
enb_save_bsln_datpck = False
enb_load_bsln_datpck = False

## enable collect baseline data using DataPack.
enb_sim_bsln_datpck = True

## ---------------------------------------------------- ##
## For Use Cases simulation.
enb_sim_usec = True 

## enable collect Use Case data using DataPack.
enb_sim_usec_datpck = True

## Enable program of Use Cases.
enb_usec_1 = False     ## Different Max Term of Loan
enb_usec_2 = True     ## Different Saving Coefficient
enb_usec_3 = True     ## Different Minimum House Price 
enb_usec_4 = False     ## Different Repayment Capability
enb_usec_5 = True     ## Different Alpha Weight
enb_usec_6 = False      ## Different LTI Limit 

## ---------------------------------------------------- ##
enb_visual_graph = True  ## Enable visualisation.
enb_save_csv = True ## Enable saving csv.

## Enable program about saving session. 
enb_save_session = True   ## to save session.
enb_load_session = False  ## to load session.   

## Enable the specific Folder Path
enb_folder_path = False

## ---------------------------------------------------- ##
n_people = 30
n_stat_round = 3
    
# =============================================================================
if enb_main == True:    
    ## initiate Simulator object
    sim = Simulator()
    
    # -------------------------------------------------------------------------
    ## initiate Visualiser object.
    vis = Visualiser()
    ## initiate UseCase object
    usec = UseCase()
    
    # -------------------------------------------------------------------------
    ## initiate Baseline scenario.
    bsln_scen = Scenario(w_alpha = 0.5, w_beta = 0.5, LTI_limit = 4.5
                     , n_dstb_sample = n_people
                     , n_brw = n_people
                     , n_bank = 10, median_ann_income = 30000
                     , min_Price_PTI_ratio = 2.5
                     , mean_saving_coef = 0.4, mean_repay_capabil = 0.4
                     , mean_max_loan_term = 300, mean_eff_growth_rate = 0.05
                     , bkst_ngenf = 5, bkst_npf = 10
                     , bkst_CR = 0.9, bkst_Ft = 0.8
                     , ir_max = 1.00, ir_min = 0.00
                     , LTV_max = 1.00, LTV_min = 0.00)
    
    # -------------------------------------------------------------------------
    ## Set the Folder Path for the picture.
    if enb_folder_path == True:
        vis.pic_filepath = "../pic/"
    else:
        vis.pic_filepath = ""
                
    # -------------------------------------------------------------------------            
    ## simulation of baseline scenario one round.
    if enb_a_sim_bsln == True:
        ## Enable to collect data using DataPack object.
        if enb_sim_bsln_datpck == True:
            sim.enb_datpck = True
        
        ## set scenario and simulate.
        sim.set_scenario(bsln_scen)
        bsln_scen, bsln_datpck = sim.simulate_a_scene(bsln_scen)
        
        ## Disable after using DataPack.
        sim.enb_datpck = False
        # -------------------------------------------------------------------------
        ## get the result of baseline scenario.
        df_bsln_scen = df_from_obj(bsln_scen)
        
        ## get the DataPack into a list.
        bsln_datpck_lst = []
        bsln_datpck_lst.append(bsln_datpck)
        
        # -------------------------------------------------------------------------
        ## visualise graphs 
        if enb_visual_graph == True:
            vis.plot_graphs(bsln_datpck_lst[0])
    
        if enb_save_bsln_datpck == True:
            ## save dataset in DataPack in a file.
            if enb_folder_path == True:
                folder_path = "../result/"
            else:
                folder_path = ""
                
            file_name = folder_path + 'bsln_datpck.pkl'
            save_object(bsln_datpck_lst[0], file_name)
            print("Saved DataPack : " + file_name)
            
        if enb_load_bsln_datpck == True:
            bsln_datpck_load = load_object(file_name)
            print("Loaded DataPack : " + file_name)
            bsln_datpck_load_lst = []
            bsln_datpck_load_lst.append(bsln_datpck_load)
    
    ## simulation of baseline scenario many rounds.
    if enb_multi_sim_bsln == True:
        ## creat a list of scenario.
        bsln_scen_lst = []
        bsln_scen_lst.append(bsln_scen)
        
        ## set number of round for multiple simulation.
        sim.stat_round = n_stat_round
        
        ## Enable to collect data using DataPack object.
        if enb_sim_bsln_datpck == True:
            sim.enb_datpck = True
        
        print("### simulate baseline scenario - multiple simulation ###")
        df_out_bsln_scen_lst, bsln_datpck_lst = sim.multiple_simulate(bsln_scen_lst)
        
        ## Disable after using DataPack.
        sim.enb_datpck = False
        
        ## --------------------------------------------------------------------
        ## visualise boxplot 
        if enb_visual_graph == True:
            vis.boxplot_baseline_res(df_out_bsln_scen_lst)
        
        ## save DataFrame df_out_scen_lst to CSV file.
        if enb_save_csv == True:
            if enb_folder_path == True:
                folder_path = "../csv/"
            else:
                folder_path = ""
            
            file_name = folder_path + "df_out_bsln_scen_lst.csv"
            df_out_bsln_scen_lst.to_csv(file_name, sep='\t'
                                        , index_label='indx'
                                        , encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_bsln_scen_lst)) + ' records')
        
    # -------------------------------------------------------------------------
    if enb_sim_usec == True:
        
        ## set number of round for multiple simulation.
        sim.stat_round = n_stat_round
        
        ## Enable to collect data using DataPack object.
        if enb_sim_usec_datpck == True:
            sim.enb_datpck = True
            
    # -------------------------------------------------------------------------  
        ## initiate many scenarios of use case.
        if enb_usec_1 == True:
            scen_lst_1 = usec.init_scenario_list(bsln_scen, usecase_set=1)
            
        if enb_usec_2 == True:
            scen_lst_2 = usec.init_scenario_list(bsln_scen, usecase_set=2)
            
        if enb_usec_3 == True:
            scen_lst_3 = usec.init_scenario_list(bsln_scen, usecase_set=3)
            
        if enb_usec_4 == True:
            scen_lst_4 = usec.init_scenario_list(bsln_scen, usecase_set=4)
            
        if enb_usec_5 == True:
            scen_lst_5 = usec.init_scenario_list(bsln_scen, usecase_set=5)
        
        if enb_usec_6 == True:
            scen_lst_6 = usec.init_scenario_list(bsln_scen, usecase_set=6)

    # -------------------------------------------------------------------------
        if enb_usec_1 == True:
            print("### simulate scen_lst_1 ###")
            df_out_scen_lst_1, datpck_lst_1 = sim.multiple_simulate(scen_lst_1)
        
        if enb_usec_2 == True:
            print("### simulate scen_lst_2 ###")
            df_out_scen_lst_2, datpck_lst_2 = sim.multiple_simulate(scen_lst_2)
        
        if enb_usec_3 == True:
            print("### simulate scen_lst_3 ###")
            df_out_scen_lst_3, datpck_lst_3 = sim.multiple_simulate(scen_lst_3)
    
        if enb_usec_4 == True:    
            print("### simulate scen_lst_4 ###")
            df_out_scen_lst_4, datpck_lst_4 = sim.multiple_simulate(scen_lst_4)
            
        if enb_usec_5 == True:
            print("### simulate scen_lst_5 ###")
            df_out_scen_lst_5, datpck_lst_5 = sim.multiple_simulate(scen_lst_5)
        
        if enb_usec_6 == True:
            print("### simulate scen_lst_6 ###")
            df_out_scen_lst_6, datpck_lst_6 = sim.multiple_simulate(scen_lst_6)
    
        ##---------------------------------------------------------------------
        ## Disable after using DataPack.
        sim.enb_datpck = False
        
        ##---------------------------------------------------------------------
        ## visualise graphs of use case.
        if enb_visual_graph == True:
            
            if enb_usec_1 == True:
                vis.plot_usecase_1_res(df_out_scen_lst_1)
                vis.boxplot_usecase_1_res(df_out_scen_lst_1)
    
            if enb_usec_2 == True:
                vis.plot_usecase_2_res(df_out_scen_lst_2)
                vis.boxplot_usecase_2_res(df_out_scen_lst_2)
                
            if enb_usec_3 == True:
                vis.plot_usecase_3_res(df_out_scen_lst_3)
                vis.boxplot_usecase_3_res(df_out_scen_lst_3)
            
            if enb_usec_4 == True:
                vis.plot_usecase_4_res(df_out_scen_lst_4)
                vis.boxplot_usecase_4_res(df_out_scen_lst_4)
                
            if enb_usec_5 == True:
                vis.plot_usecase_5_res(df_out_scen_lst_5)
                vis.boxplot_usecase_5_res(df_out_scen_lst_5)
                
            if enb_usec_6 == True:
                vis.plot_usecase_6_res(df_out_scen_lst_6)
                vis.boxplot_usecase_6_res(df_out_scen_lst_6)
        
        ##---------------------------------------------------------------------
        ## save DataFrame df_out_scen_lst to CSV file.
        if enb_folder_path == True:
            folder_path = "../csv/"
        else:
            folder_path = ""
        
        if enb_usec_1 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_1.csv"
            df_out_scen_lst_1.to_csv(file_name
                                     , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_1)) + ' records')
        
        if enb_usec_2 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_2.csv"
            df_out_scen_lst_2.to_csv(file_name
                                     , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_2)) + ' records')
         
        if enb_usec_3 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_3.csv"
            df_out_scen_lst_3.to_csv(file_name
                                 , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_3)) + ' records')
            
        if enb_usec_4 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_4.csv"
            df_out_scen_lst_4.to_csv(file_name
                                     , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_4)) + ' records')
         
        if enb_usec_5 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_5.csv"
            df_out_scen_lst_5.to_csv(file_name
                                     , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_5)) + ' records')
            
        if enb_usec_6 == True and enb_save_csv == True:
            file_name = folder_path + "df_out_scen_lst_6.csv"
            df_out_scen_lst_6.to_csv(file_name
                                     , sep='\t', index_label='indx', encoding='utf-8')
            print( 'saved txt file : ' + str(len(df_out_scen_lst_6)) + ' records')
                
## ----------------------------------------------------------------------------
    
# =============================================================================
# Save python session after finish simulation.  
# =============================================================================
## define path and file name to be saved.     
if enb_folder_path == True:
    folder_path = "../session/"
else:
    folder_path = ""  
    
filename = folder_path + "mgt_game_04vv_kk_session.pkl"    

# -----------------------------------------------------------------------------
# save the session using dill
# -----------------------------------------------------------------------------
if enb_save_session == True:
    dill.dump_session(filename)
    print("-------------------------------------------------------")
    print("save session : "+str(filename))
    print("-------------------------------------------------------")

# -----------------------------------------------------------------------------
# load the session using dill
# -----------------------------------------------------------------------------
if enb_load_session == True:
    dill.load_session(filename)
    print("-------------------------------------------------------")
    print("load session : "+str(filename))
    print("-------------------------------------------------------")
    
# *****************************************************************************
# 
# *****************************************************************************
# -----------------------------------------------------------------------------
# Show the finish time of this program.
# -----------------------------------------------------------------------------
get_dt_finish_time = dt.datetime.now()
print("-------------------------------------------------------")
print("finish time : "+str(get_dt_finish_time))
print("-------------------------------------------------------")
# -----------------------------------------------------------------------------
# Show the amount of time to run this program.
# -----------------------------------------------------------------------------
get_dt_running_time = get_dt_finish_time - get_dt_start_time
print("-------------------------------------------------------")
print("running time : "+str(get_dt_running_time))
print("-------------------------------------------------------")
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------

    