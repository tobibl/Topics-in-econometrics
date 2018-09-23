#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:11:12 2018

@author: owner
"""



import requests as rq
import pandas as pd
import numpy as np
import os 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

###import 1 month treasury bill rate monthly
os.getcwd()
os.chdir(r'/users/owner/Desktop')

risk_free_rate = pd.read_csv('DGS1MO.csv',index_col=0)

###############################################################################
#########  Calling Aplha Vantage API ##########################################
###############################################################################

#URL
API_URL = "https://www.alphavantage.co/query"


def get_stock(stock_abrv):
    params = {'function':'TIME_SERIES_MONTHLY','symbol':stock_abrv,'outputsize':'full','apikey':'W34JWX35ZWNFUA0D'}
    response = rq.get(API_URL,params)
    data=response.json()
    #Second list contains values
    data1 = list(data.values())[1]
    
    
    return pd.DataFrame.from_dict(data1,orient='index').convert_objects(convert_numeric=True)
    

nasdaq = get_stock('^IXIC')
microsoft = get_stock('MSFT')
amazon = get_stock('AMZN')
apple = get_stock('AAPL')
qualcomm = get_stock ('QCOM')
intel = get_stock('INTC')
nvidia = get_stock('NVDA')
comcast = get_stock('CMCSA')
cisco = get_stock('CSCO')
ebay = get_stock('EBAY')
starbucks = get_stock('SBUX')



###############################################################################
############ create monthly log returns #######################################
###############################################################################

riskfree_log_returns = np.log(risk_free_rate / risk_free_rate.shift(-1))

apple_log_returns = np.log(apple / apple.shift(-1))

amazon_log_returns = np.log(amazon / amazon.shift(-1))

microsoft_log_returns = np.log(microsoft / microsoft.shift(-1))

nasdaq_log_returns = np.log(nasdaq / nasdaq.shift(-1))

cisco_log_returns= np.log(cisco / cisco.shift(-1))

comcast_log_returns= np.log(comcast / comcast.shift(-1))

ebay_log_returns= np.log(ebay / ebay.shift(-1))

intel_log_returns= np.log(intel / intel.shift(-1))

nvidia_log_returns= np.log(nvidia / nvidia.shift(-1))

qualcomm_log_returns= np.log(qualcomm / qualcomm.shift(-1))

starbucks_log_returns= np.log(starbucks / starbucks.shift(-1))

###############################################################################
####bring all data frames to the same date length since my risk_free data first starts in september 2001
###############################################################################


riskfree_log_returns.drop(riskfree_log_returns.tail(1).index, inplace=True)


apple_log_returns.drop(apple_log_returns.head(78).index,inplace=True)
apple_log_returns.drop(apple_log_returns.tail(4).index,inplace=True)

amazon_log_returns.drop(amazon_log_returns.head(50).index, inplace=True)
amazon_log_returns.drop(amazon_log_returns.tail(4).index, inplace=True)

microsoft_log_returns.drop(microsoft_log_returns.head(78).index, inplace=True)
microsoft_log_returns.drop(microsoft_log_returns.tail(4).index, inplace=True)

nasdaq_log_returns.drop(nasdaq_log_returns.head(18).index, inplace=True)
nasdaq_log_returns.drop(nasdaq_log_returns.tail(4).index, inplace=True)

cisco_log_returns.drop(cisco_log_returns.head(78).index, inplace=True)
cisco_log_returns.drop(cisco_log_returns.tail(4).index, inplace=True)

comcast_log_returns.drop(comcast_log_returns.head(78).index, inplace=True)
comcast_log_returns.drop(comcast_log_returns.tail(4).index, inplace=True)

ebay_log_returns.drop(ebay_log_returns.head(34).index, inplace=True)
ebay_log_returns.drop(ebay_log_returns.tail(4).index, inplace=True)

intel_log_returns.drop(intel_log_returns.head(78).index, inplace=True)
intel_log_returns.drop(intel_log_returns.tail(4).index, inplace=True)

nvidia_log_returns.drop(nvidia_log_returns.head(30).index, inplace=True)
nvidia_log_returns.drop(nvidia_log_returns.tail(4).index, inplace=True)

qualcomm_log_returns.drop(qualcomm_log_returns.head(78).index, inplace=True)
qualcomm_log_returns.drop(qualcomm_log_returns.tail(4).index, inplace=True)

starbucks_log_returns.drop(starbucks_log_returns.head(78).index, inplace=True)
starbucks_log_returns.drop(starbucks_log_returns.tail(4).index, inplace=True)

###############################################################################
####adjusting date index to month/Year for easier matrix calculation###########
###############################################################################

apple_log_returns.index = pd.to_datetime(apple_log_returns.index).strftime('%m-%Y')
riskfree_log_returns.index = pd.to_datetime(riskfree_log_returns.index).strftime('%m-%Y')
nasdaq_log_returns.index = pd.to_datetime(nasdaq_log_returns.index).strftime('%m-%Y')
microsoft_log_returns.index = pd.to_datetime(microsoft_log_returns.index).strftime('%m-%Y')
amazon_log_returns.index = pd.to_datetime(amazon_log_returns.index ).strftime('%m-%Y')

cisco_log_returns.index = pd.to_datetime(cisco_log_returns.index ).strftime('%m-%Y')
comcast_log_returns.index = pd.to_datetime(comcast_log_returns.index ).strftime('%m-%Y')
ebay_log_returns.index = pd.to_datetime(ebay_log_returns.index ).strftime('%m-%Y')
intel_log_returns.index = pd.to_datetime(intel_log_returns.index ).strftime('%m-%Y')
nvidia_log_returns.index = pd.to_datetime(nvidia_log_returns.index ).strftime('%m-%Y')
qualcomm_log_returns.index = pd.to_datetime(qualcomm_log_returns.index ).strftime('%m-%Y')
starbucks_log_returns.index = pd.to_datetime(starbucks_log_returns.index ).strftime('%m-%Y')


###############################################################################
#######create excess returns (er) for assets and nasdaq of closing price#######
###############################################################################


apple_er = apple_log_returns['4. close'].values - riskfree_log_returns.DGS1MO.values
amazon_er = amazon_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
microsoft_er = microsoft_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
nasdaq_er = nasdaq_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values

cisco_er = cisco_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
comcast_er = comcast_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
ebay_er = ebay_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
intel_er = intel_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
nvidia_er = nvidia_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
qualcomm_er = qualcomm_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values
starbucks_er = starbucks_log_returns['4. close'].values- riskfree_log_returns.DGS1MO.values


###############################################################################
######OLS Regression for CAPM alpha and beta###################################
###############################################################################

results_apple = smf.ols(formula= 'apple_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_apple.summary())

results_amazon = smf.ols(formula= 'amazon_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_amazon.summary())

results_microsoft = smf.ols(formula= 'microsoft_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_microsoft.summary())


results_cisco = smf.ols(formula= 'cisco_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_cisco.summary())

results_comcast = smf.ols(formula= 'comcast_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_comcast.summary())

results_ebay = smf.ols(formula= 'ebay_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_ebay.summary())

results_intel = smf.ols(formula= 'intel_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_intel.summary())

results_nvidia = smf.ols(formula= 'nvidia_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_nvidia.summary())

results_qualcomm = smf.ols(formula= 'qualcomm_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_qualcomm.summary())

results_starbucks = smf.ols(formula= 'starbucks_er ~ nasdaq_er', data=nasdaq_log_returns).fit()
print(results_starbucks.summary())


###############################################################################
####Use estimated residuals of residuals for each asset to compute sigma_hat #####
###############################################################################

####apple##
predictedValues = results_apple.predict()

res_apple = np.array([apple_er - predictedValues])
res_apple.shape


apple_res_transpose=np.transpose(res_apple)

apple_sigma=(1/202)*np.matmul(res_apple , apple_res_transpose)
print(apple_sigma)



####amazon##
predictedValues_amazon = results_amazon.predict()

res_amazon = np.array([amazon_er - predictedValues_amazon])



amazon_res_transpose=np.transpose(res_amazon)

amazon_sigma=(1/202)*np.matmul(res_amazon , amazon_res_transpose)
print(amazon_sigma)


####microsoft##
predictedValues_microsoft = results_microsoft.predict()

res_microsoft = np.array([microsoft_er - predictedValues_microsoft])
res_microsoft.shape


microsoft_res_transpose=np.transpose(res_microsoft)

microsoft_sigma=(1/202)*np.matmul(res_microsoft , microsoft_res_transpose)
print(microsoft_sigma)


####cisco##
predictedValues_cisco = results_cisco.predict()

res_cisco = np.array([cisco_er - predictedValues_cisco])


cisco_res_transpose=np.transpose(res_cisco)

cisco_sigma=(1/202)*np.matmul(res_cisco , cisco_res_transpose)
print(cisco_sigma)


####comcast##
predictedValues_comcast = results_comcast.predict()

res_comcast = np.array([comcast_er - predictedValues_comcast])


comcast_res_transpose=np.transpose(res_comcast)

comcast_sigma=(1/202)*np.matmul(res_comcast , comcast_res_transpose)
print(comcast_sigma)


####ebay##
predictedValues_ebay = results_ebay.predict()

res_ebay = np.array([ebay_er - predictedValues_ebay])


ebay_res_transpose=np.transpose(res_ebay)

ebay_sigma=(1/202)*np.matmul(res_ebay , ebay_res_transpose)
print(ebay_sigma)


####intel##
predictedValues_intel = results_intel.predict()

res_intel = np.array([intel_er - predictedValues_intel])


intel_res_transpose=np.transpose(res_intel)

intel_sigma=(1/202)*np.matmul(res_intel, intel_res_transpose)
print(intel_sigma)


####nvidia##
predictedValues_nvidia = results_nvidia.predict()

res_nvidia = np.array([nvidia_er - predictedValues_nvidia])


nvidia_res_transpose=np.transpose(res_nvidia)

nvidia_sigma=(1/202)*np.matmul(res_nvidia , nvidia_res_transpose)
print(nvidia_sigma)


####qualcomm##
predictedValues_qualcomm = results_qualcomm.predict()

res_qualcomm = np.array([qualcomm_er - predictedValues_qualcomm])


qualcomm_res_transpose=np.transpose(res_qualcomm)

qualcomm_sigma=(1/202)*np.matmul(res_qualcomm , qualcomm_res_transpose)
print(qualcomm_sigma)


####starbucks##
predictedValues_starbucks = results_starbucks.predict()

res_starbucks = np.array([starbucks_er - predictedValues_starbucks])


starbucks_res_transpose=np.transpose(res_starbucks)

starbucks_sigma=(1/202)*np.matmul(res_starbucks ,starbucks_res_transpose)
print(starbucks_sigma)

###############################################################################
####get alpha values for each stock############################################
###############################################################################


apple_alpha = results_apple.params['Intercept']
amazon_alpha = results_amazon.params['Intercept']
cisco_alpha = results_cisco.params['Intercept']
comcast_alpha = results_comcast.params['Intercept']
ebay_alpha = results_ebay.params['Intercept']
intel_alpha = results_intel.params['Intercept']
microsoft_alpha = results_microsoft.params['Intercept']
nvidia_alpha = results_nvidia.params['Intercept']
qualcomm_alpha = results_qualcomm.params['Intercept']
starbucks_alpha = results_starbucks.params['Intercept']

###############################################################################
#####Compute (alpha'*Sigma^-1*alpha) as _stat for each asset and sum of all####
###############################################################################


apple_stat = apple_alpha*(apple_sigma**-1)*apple_alpha
amazon_stat = amazon_alpha*(amazon_sigma**-1)*amazon_alpha
cisco_stat = cisco_alpha*(cisco_sigma**-1)*cisco_alpha
comcast_stat = comcast_alpha*(comcast_sigma**-1)*comcast_alpha
ebay_stat = ebay_alpha*(ebay_sigma**-1)*ebay_alpha
intel_stat = intel_alpha*(intel_sigma**-1)*intel_alpha
microsoft_stat = microsoft_alpha*(microsoft_sigma**-1)*microsoft_alpha
nvidia_stat = nvidia_alpha*(nvidia_sigma**-1)*nvidia_alpha
qualcomm_stat = qualcomm_alpha*(qualcomm_sigma**-1)*qualcomm_alpha
starbucks_stat = starbucks_alpha*(starbucks_sigma**-1)*starbucks_alpha

asset_sum = apple_stat + amazon_stat + cisco_stat+ebay_stat+intel_stat+microsoft_stat+nvidia_stat+qualcomm_stat+starbucks_stat
print(asset_sum)

###############################################################################
####Calculate squared mean and std deviation sigma-squared for nasdaq##########
###############################################################################


mu_nasdaq = np.mean(nasdaq_er)
std_nasdaq = np.std(nasdaq_er)



###############################################################################
#######Calculate Wald test statistic###########################################
###############################################################################


w= 202*(1 + mu_nasdaq**2 / std_nasdaq)**-1 * asset_sum
print(w)


###############################################################################
####Calculate F-statistic J_1##################################################
###############################################################################


J_1 = (202-10-1)/10 *(1 + mu_nasdaq**2 / std_nasdaq)**-1 * asset_sum
print(J_1)


####Calculate the CAPM F test for economic interpretation###



###############################################################################
###############################################################################
##### divide portfolio first subperiod 08/2001 - 08/2005   ####################
###############################################################################
###############################################################################

nasdaq1_log_returns = nasdaq_log_returns[0:48]
nasdaq1_er=nasdaq_er[0:48]

amazon1_er=amazon_er[0:48]
apple1_er=apple_er[0:48]
microsoft1_er=microsoft_er[0:48]
cisco1_er=cisco_er[0:48]
comcast1_er=comcast_er[0:48]
ebay1_er=ebay_er[0:48]
intel1_er=intel_er[0:48]
nvidia1_er=nvidia_er[0:48]
qualcomm1_er=qualcomm_er[0:48]
starbucks1_er= starbucks_er[0:48]


###############################################################################
######OLS Regression for CAPM alpha and apple11 beta###########################
###############################################################################


results_apple1 = smf.ols(formula= 'apple1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_apple1.summary())
results_amazon1 = smf.ols(formula= 'amazon1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_amazon1.summary())
results_microsoft1 = smf.ols(formula= 'microsoft1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_microsoft1.summary())

results_cisco1 = smf.ols(formula= 'cisco1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_cisco1.summary())
results_comcast1 = smf.ols(formula= 'comcast1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_comcast1.summary())
results_ebay1 = smf.ols(formula= 'ebay1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_ebay1.summary())
results_intel1 = smf.ols(formula= 'intel1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_intel1.summary())
results_nvidia1 = smf.ols(formula= 'nvidia1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_nvidia1.summary())
results_qualcomm1 = smf.ols(formula= 'qualcomm1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_qualcomm1.summary())
results_starbucks1 = smf.ols(formula= 'starbucks1_er ~ nasdaq1_er', data=nasdaq1_log_returns).fit()
print(results_starbucks1.summary())

###############################################################################
####Use estimated residuals of residuals for each asset to compute sigma_hat #####
###############################################################################
####apple1##
predictedValues = results_apple1.predict()
res_apple1 = np.array([apple1_er - predictedValues])
res_apple1.shape

apple1_res_transpose=np.transpose(res_apple1)
apple1_sigma=(1/202)*np.matmul(res_apple1 , apple1_res_transpose)
print(apple1_sigma)

####amazon1##
predictedValues_amazon1 = results_amazon1.predict()
res_amazon1 = np.array([amazon1_er - predictedValues_amazon1])

amazon1_res_transpose=np.transpose(res_amazon1)
amazon1_sigma=(1/202)*np.matmul(res_amazon1 , amazon1_res_transpose)
print(amazon1_sigma)

####microsoft1##
predictedValues_microsoft1 = results_microsoft1.predict()
res_microsoft1 = np.array([microsoft1_er - predictedValues_microsoft1])
res_microsoft1.shape

microsoft1_res_transpose=np.transpose(res_microsoft1)
microsoft1_sigma=(1/202)*np.matmul(res_microsoft1 , microsoft1_res_transpose)
print(microsoft1_sigma)

####cisco1##
predictedValues_cisco1 = results_cisco1.predict()
res_cisco1 = np.array([cisco1_er - predictedValues_cisco1])

cisco1_res_transpose=np.transpose(res_cisco1)
cisco1_sigma=(1/202)*np.matmul(res_cisco1 , cisco1_res_transpose)
print(cisco1_sigma)

####comcast1##
predictedValues_comcast1 = results_comcast1.predict()
res_comcast1 = np.array([comcast1_er - predictedValues_comcast1])

comcast1_res_transpose=np.transpose(res_comcast1)
comcast1_sigma=(1/202)*np.matmul(res_comcast1 , comcast1_res_transpose)
print(comcast1_sigma)

####ebay1##
predictedValues_ebay1 = results_ebay1.predict()
res_ebay1 = np.array([ebay1_er - predictedValues_ebay1])

ebay1_res_transpose=np.transpose(res_ebay1)
ebay1_sigma=(1/202)*np.matmul(res_ebay1 , ebay1_res_transpose)
print(ebay1_sigma)

####intel1##
predictedValues_intel1 = results_intel1.predict()
res_intel1 = np.array([intel1_er - predictedValues_intel1])

intel1_res_transpose=np.transpose(res_intel1)
intel1_sigma=(1/202)*np.matmul(res_intel1, intel1_res_transpose)
print(intel1_sigma)

####nvidia1##
predictedValues_nvidia1 = results_nvidia1.predict()
res_nvidia1 = np.array([nvidia1_er - predictedValues_nvidia1])

nvidia1_res_transpose=np.transpose(res_nvidia1)
nvidia1_sigma=(1/202)*np.matmul(res_nvidia1 , nvidia1_res_transpose)
print(nvidia1_sigma)

####qualcomm1##
predictedValues_qualcomm1 = results_qualcomm1.predict()
res_qualcomm1 = np.array([qualcomm1_er - predictedValues_qualcomm1])

qualcomm1_res_transpose=np.transpose(res_qualcomm1)
qualcomm1_sigma=(1/202)*np.matmul(res_qualcomm1 , qualcomm1_res_transpose)
print(qualcomm1_sigma)

####starbucks1##
predictedValues_starbucks1 = results_starbucks1.predict()
res_starbucks1 = np.array([starbucks1_er - predictedValues_starbucks1])

starbucks1_res_transpose=np.transpose(res_starbucks1)
starbucks1_sigma=(1/202)*np.matmul(res_starbucks1 ,starbucks1_res_transpose)
print(starbucks1_sigma)
###############################################################################
####get alpha values for each stock############################################
###############################################################################

apple1_alpha = results_apple1.params['Intercept']
amazon1_alpha = results_amazon1.params['Intercept']
cisco1_alpha = results_cisco1.params['Intercept']
comcast1_alpha = results_comcast1.params['Intercept']
ebay1_alpha = results_ebay1.params['Intercept']
intel1_alpha = results_intel1.params['Intercept']
microsoft1_alpha = results_microsoft1.params['Intercept']
nvidia1_alpha = results_nvidia1.params['Intercept']
qualcomm1_alpha = results_qualcomm1.params['Intercept']
starbucks1_alpha = results_starbucks1.params['Intercept']
###############################################################################
#####Compute (alpha'*Sigma^-1*alpha) as _stat for each asset and sum of all####
###############################################################################

apple1_stat = apple1_alpha*(apple1_sigma**-1)*apple1_alpha
amazon1_stat = amazon1_alpha*(amazon1_sigma**-1)*amazon1_alpha
cisco1_stat = cisco1_alpha*(cisco1_sigma**-1)*cisco1_alpha
comcast1_stat = comcast1_alpha*(comcast1_sigma**-1)*comcast1_alpha
ebay1_stat = ebay1_alpha*(ebay1_sigma**-1)*ebay1_alpha
intel1_stat = intel1_alpha*(intel1_sigma**-1)*intel1_alpha
microsoft1_stat = microsoft1_alpha*(microsoft1_sigma**-1)*microsoft1_alpha
nvidia1_stat = nvidia1_alpha*(nvidia1_sigma**-1)*nvidia1_alpha
qualcomm1_stat = qualcomm1_alpha*(qualcomm1_sigma**-1)*qualcomm1_alpha
starbucks1_stat = starbucks1_alpha*(starbucks1_sigma**-1)*starbucks1_alpha
asset1_sum = apple1_stat + amazon1_stat + cisco1_stat+ebay1_stat+intel1_stat+microsoft1_stat+nvidia1_stat+qualcomm1_stat+starbucks1_stat
print(asset1_sum)


###############################################################################
####Calculate squared mean and std deviation sigma-squared for nasdaq##########
###############################################################################


mu_nasdaq1 = np.mean(nasdaq1_er)
std_nasdaq1 = np.std(nasdaq1_er)



###############################################################################
#######Calculate Wald test statistic###########################################
###############################################################################


w1 = 48*(1 + mu_nasdaq1**2 / std_nasdaq1)**-1 * asset1_sum
print(w1)


###############################################################################
####Calculate F-statistic J_1##################################################
###############################################################################


J_11 = (48-10-1)/10 *(1 + mu_nasdaq1**2 / std_nasdaq1)**-1 * asset1_sum
print(J_11)


####Calculate the CAPM F test for economic interpretation###

###############################################################################
###############################################################################
##### divide portfolio second subperiod 09/2005 - 09/2009   ####################
###############################################################################
###############################################################################



nasdaq2_log_returns = nasdaq_log_returns[49:97]
nasdaq2_er=nasdaq_er[49:97]

amazon2_er=amazon_er[49:97]
apple2_er=apple_er[49:97]
microsoft2_er=microsoft_er[49:97]
cisco2_er=cisco_er[49:97]
comcast2_er=comcast_er[49:97]
ebay2_er=ebay_er[49:97]
intel2_er=intel_er[49:97]
nvidia2_er=nvidia_er[49:97]
qualcomm2_er=qualcomm_er[49:97]
starbucks2_er= starbucks_er[49:97]





######OLS Regression for CAPM alpha and beta###################################
###############################################################################
results_apple2 = smf.ols(formula= 'apple2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_apple2.summary())
results_amazon2 = smf.ols(formula= 'amazon2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_amazon2.summary())
results_microsoft2 = smf.ols(formula= 'microsoft2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_microsoft2.summary())

results_cisco2 = smf.ols(formula= 'cisco2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_cisco2.summary())
results_comcast2 = smf.ols(formula= 'comcast2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_comcast2.summary())
results_ebay2 = smf.ols(formula= 'ebay2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_ebay2.summary())
results_intel2 = smf.ols(formula= 'intel2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_intel2.summary())
results_nvidia2 = smf.ols(formula= 'nvidia2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_nvidia2.summary())
results_qualcomm2 = smf.ols(formula= 'qualcomm2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_qualcomm2.summary())
results_starbucks2 = smf.ols(formula= 'starbucks2_er ~ nasdaq2_er', data=nasdaq2_log_returns).fit()
print(results_starbucks2.summary())

###############################################################################
####Use estimated residuals of residuals for each asset to compute sigma_hat #####
###############################################################################
####apple2##
predictedValues = results_apple2.predict()
res_apple2 = np.array([apple2_er - predictedValues])
res_apple2.shape

apple2_res_transpose=np.transpose(res_apple2)
apple2_sigma=(1/202)*np.matmul(res_apple2 , apple2_res_transpose)
print(apple2_sigma)

####amazon2##
predictedValues_amazon2 = results_amazon2.predict()
res_amazon2 = np.array([amazon2_er - predictedValues_amazon2])

amazon2_res_transpose=np.transpose(res_amazon2)
amazon2_sigma=(1/202)*np.matmul(res_amazon2 , amazon2_res_transpose)
print(amazon2_sigma)

####microsoft2##
predictedValues_microsoft2 = results_microsoft2.predict()
res_microsoft2 = np.array([microsoft2_er - predictedValues_microsoft2])
res_microsoft2.shape

microsoft2_res_transpose=np.transpose(res_microsoft2)
microsoft2_sigma=(1/202)*np.matmul(res_microsoft2 , microsoft2_res_transpose)
print(microsoft2_sigma)

####cisco2##
predictedValues_cisco2 = results_cisco2.predict()
res_cisco2 = np.array([cisco2_er - predictedValues_cisco2])

cisco2_res_transpose=np.transpose(res_cisco2)
cisco2_sigma=(1/202)*np.matmul(res_cisco2 , cisco2_res_transpose)
print(cisco2_sigma)

####comcast2##
predictedValues_comcast2 = results_comcast2.predict()
res_comcast2 = np.array([comcast2_er - predictedValues_comcast2])

comcast2_res_transpose=np.transpose(res_comcast2)
comcast2_sigma=(1/202)*np.matmul(res_comcast2 , comcast2_res_transpose)
print(comcast2_sigma)

####ebay2##
predictedValues_ebay2 = results_ebay2.predict()
res_ebay2 = np.array([ebay2_er - predictedValues_ebay2])

ebay2_res_transpose=np.transpose(res_ebay2)
ebay2_sigma=(1/202)*np.matmul(res_ebay2 , ebay2_res_transpose)
print(ebay2_sigma)

####intel2##
predictedValues_intel2 = results_intel2.predict()
res_intel2 = np.array([intel2_er - predictedValues_intel2])

intel2_res_transpose=np.transpose(res_intel2)
intel2_sigma=(1/202)*np.matmul(res_intel2, intel2_res_transpose)
print(intel2_sigma)

####nvidia2##
predictedValues_nvidia2 = results_nvidia2.predict()
res_nvidia2 = np.array([nvidia2_er - predictedValues_nvidia2])

nvidia2_res_transpose=np.transpose(res_nvidia2)
nvidia2_sigma=(1/202)*np.matmul(res_nvidia2 , nvidia2_res_transpose)
print(nvidia2_sigma)

####qualcomm2##
predictedValues_qualcomm2 = results_qualcomm2.predict()
res_qualcomm2 = np.array([qualcomm2_er - predictedValues_qualcomm2])

qualcomm2_res_transpose=np.transpose(res_qualcomm2)
qualcomm2_sigma=(1/202)*np.matmul(res_qualcomm2 , qualcomm2_res_transpose)
print(qualcomm2_sigma)

####starbucks2##
predictedValues_starbucks2 = results_starbucks2.predict()
res_starbucks2 = np.array([starbucks2_er - predictedValues_starbucks2])

starbucks2_res_transpose=np.transpose(res_starbucks2)
starbucks2_sigma=(1/202)*np.matmul(res_starbucks2 ,starbucks2_res_transpose)
print(starbucks2_sigma)
###############################################################################
####get alpha values for each stock############################################
###############################################################################

apple2_alpha = results_apple2.params['Intercept']
amazon2_alpha = results_amazon2.params['Intercept']
cisco2_alpha = results_cisco2.params['Intercept']
comcast2_alpha = results_comcast2.params['Intercept']
ebay2_alpha = results_ebay2.params['Intercept']
intel2_alpha = results_intel2.params['Intercept']
microsoft2_alpha = results_microsoft2.params['Intercept']
nvidia2_alpha = results_nvidia2.params['Intercept']
qualcomm2_alpha = results_qualcomm2.params['Intercept']
starbucks2_alpha = results_starbucks2.params['Intercept']
###############################################################################
#####Compute (alpha'*Sigma^-1*alpha) as _stat for each asset and sum of all####
###############################################################################

apple2_stat = apple2_alpha*(apple2_sigma**-1)*apple2_alpha
amazon2_stat = amazon2_alpha*(amazon2_sigma**-1)*amazon2_alpha
cisco2_stat = cisco2_alpha*(cisco2_sigma**-1)*cisco2_alpha
comcast2_stat = comcast2_alpha*(comcast2_sigma**-1)*comcast2_alpha
ebay2_stat = ebay2_alpha*(ebay2_sigma**-1)*ebay2_alpha
intel2_stat = intel2_alpha*(intel2_sigma**-1)*intel2_alpha
microsoft2_stat = microsoft2_alpha*(microsoft2_sigma**-1)*microsoft2_alpha
nvidia2_stat = nvidia2_alpha*(nvidia2_sigma**-1)*nvidia2_alpha
qualcomm2_stat = qualcomm2_alpha*(qualcomm2_sigma**-1)*qualcomm2_alpha
starbucks2_stat = starbucks2_alpha*(starbucks2_sigma**-1)*starbucks2_alpha
asset2_sum = apple2_stat + amazon2_stat + cisco2_stat+ebay2_stat+intel2_stat+microsoft2_stat+nvidia2_stat+qualcomm2_stat+starbucks2_stat
print(asset2_sum)


###############################################################################
####Calculate squared mean and std deviation sigma-squared for nasdaq##########
###############################################################################


mu_nasdaq2 = np.mean(nasdaq2_er)
std_nasdaq2 = np.std(nasdaq2_er)



###############################################################################
#######Calculate Wald test statistic###########################################
###############################################################################


w2 = 48*(1 + mu_nasdaq2**2 / std_nasdaq2)**-1 * asset2_sum
print(w2)


###############################################################################
####Calculate F-statistic J_1##################################################
###############################################################################


J_12 = (48-10-1)/10 *(1 + mu_nasdaq2**2 / std_nasdaq2)**-1 * asset2_sum
print(J_12)


####Calculate the CAPM F test for economic interpretation###


###############################################################################
###############################################################################
##### divide portfolio third subperiod 10/2009 - 10/2013   ###################
###############################################################################
###############################################################################



nasdaq3_log_returns = nasdaq_log_returns[98:146]
nasdaq3_er=nasdaq_er[98:146]

amazone3_er=amazon_er[98:146]
apple3_er=apple_er[98:146]
microsoft3_er=microsoft_er[98:146]
cisco3_er=cisco_er[98:146]
comcast3_er=comcast_er[98:146]
ebay3_er=ebay_er[98:146]
intel3_er=intel_er[98:146]
nvidia3_er=nvidia_er[98:146]
qualcomm3_er=qualcomm_er[98:146]
starbucks3_er= starbucks_er[98:146]



######OLS Regression for CAPM alpha and beta###################################
###############################################################################
results_apple3 = smf.ols(formula= 'apple3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_apple3.summary())
results_amazone3 = smf.ols(formula= 'amazone3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_amazone3.summary())
results_microsoft3 = smf.ols(formula= 'microsoft3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_microsoft3.summary())

results_cisco3 = smf.ols(formula= 'cisco3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_cisco3.summary())
results_comcast3 = smf.ols(formula= 'comcast3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_comcast3.summary())
results_ebay3 = smf.ols(formula= 'ebay3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_ebay3.summary())
results_intel3 = smf.ols(formula= 'intel3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_intel3.summary())
results_nvidia3 = smf.ols(formula= 'nvidia3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_nvidia3.summary())
results_qualcomm3 = smf.ols(formula= 'qualcomm3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_qualcomm3.summary())
results_starbucks3 = smf.ols(formula= 'starbucks3_er ~ nasdaq3_er', data=nasdaq3_log_returns).fit()
print(results_starbucks3.summary())

###############################################################################
####Use estimated residuals of residuals for each asset to compute sigma_hat #####
###############################################################################
####apple3##
predictedValues = results_apple3.predict()
res_apple3 = np.array([apple3_er - predictedValues])
res_apple3.shape

apple3_res_transpose=np.transpose(res_apple3)
apple3_sigma=(1/202)*np.matmul(res_apple3 , apple3_res_transpose)
print(apple3_sigma)

####amazone3##
predictedValues_amazone3 = results_amazone3.predict()
res_amazone3 = np.array([amazone3_er - predictedValues_amazone3])

amazone3_res_transpose=np.transpose(res_amazone3)
amazone3_sigma=(1/202)*np.matmul(res_amazone3 , amazone3_res_transpose)
print(amazone3_sigma)

####microsoft3##
predictedValues_microsoft3 = results_microsoft3.predict()
res_microsoft3 = np.array([microsoft3_er - predictedValues_microsoft3])
res_microsoft3.shape

microsoft3_res_transpose=np.transpose(res_microsoft3)
microsoft3_sigma=(1/202)*np.matmul(res_microsoft3 , microsoft3_res_transpose)
print(microsoft3_sigma)

####cisco3##
predictedValues_cisco3 = results_cisco3.predict()
res_cisco3 = np.array([cisco3_er - predictedValues_cisco3])

cisco3_res_transpose=np.transpose(res_cisco3)
cisco3_sigma=(1/202)*np.matmul(res_cisco3 , cisco3_res_transpose)
print(cisco3_sigma)

####comcast3##
predictedValues_comcast3 = results_comcast3.predict()
res_comcast3 = np.array([comcast3_er - predictedValues_comcast3])

comcast3_res_transpose=np.transpose(res_comcast3)
comcast3_sigma=(1/202)*np.matmul(res_comcast3 , comcast3_res_transpose)
print(comcast3_sigma)

####ebay3##
predictedValues_ebay3 = results_ebay3.predict()
res_ebay3 = np.array([ebay3_er - predictedValues_ebay3])

ebay3_res_transpose=np.transpose(res_ebay3)
ebay3_sigma=(1/202)*np.matmul(res_ebay3 , ebay3_res_transpose)
print(ebay3_sigma)

####intel3##
predictedValues_intel3 = results_intel3.predict()
res_intel3 = np.array([intel3_er - predictedValues_intel3])

intel3_res_transpose=np.transpose(res_intel3)
intel3_sigma=(1/202)*np.matmul(res_intel3, intel3_res_transpose)
print(intel3_sigma)

####nvidia3##
predictedValues_nvidia3 = results_nvidia3.predict()
res_nvidia3 = np.array([nvidia3_er - predictedValues_nvidia3])

nvidia3_res_transpose=np.transpose(res_nvidia3)
nvidia3_sigma=(1/202)*np.matmul(res_nvidia3 , nvidia3_res_transpose)
print(nvidia3_sigma)

####qualcomm3##
predictedValues_qualcomm3 = results_qualcomm3.predict()
res_qualcomm3 = np.array([qualcomm3_er - predictedValues_qualcomm3])

qualcomm3_res_transpose=np.transpose(res_qualcomm3)
qualcomm3_sigma=(1/202)*np.matmul(res_qualcomm3 , qualcomm3_res_transpose)
print(qualcomm3_sigma)

####starbucks3##
predictedValues_starbucks3 = results_starbucks3.predict()
res_starbucks3 = np.array([starbucks3_er - predictedValues_starbucks3])

starbucks3_res_transpose=np.transpose(res_starbucks3)
starbucks3_sigma=(1/202)*np.matmul(res_starbucks3 ,starbucks3_res_transpose)
print(starbucks3_sigma)
###############################################################################
####get alpha values for each stock############################################
###############################################################################

apple3_alpha = results_apple3.params['Intercept']
amazone3_alpha = results_amazone3.params['Intercept']
cisco3_alpha = results_cisco3.params['Intercept']
comcast3_alpha = results_comcast3.params['Intercept']
ebay3_alpha = results_ebay3.params['Intercept']
intel3_alpha = results_intel3.params['Intercept']
microsoft3_alpha = results_microsoft3.params['Intercept']
nvidia3_alpha = results_nvidia3.params['Intercept']
qualcomm3_alpha = results_qualcomm3.params['Intercept']
starbucks3_alpha = results_starbucks3.params['Intercept']
###############################################################################
#####Compute (alpha'*Sigma^-1*alpha) as _stat for each asset and sum of all####
###############################################################################

apple3_stat = apple3_alpha*(apple3_sigma**-1)*apple3_alpha
amazone3_stat = amazone3_alpha*(amazone3_sigma**-1)*amazone3_alpha
cisco3_stat = cisco3_alpha*(cisco3_sigma**-1)*cisco3_alpha
comcast3_stat = comcast3_alpha*(comcast3_sigma**-1)*comcast3_alpha
ebay3_stat = ebay3_alpha*(ebay3_sigma**-1)*ebay3_alpha
intel3_stat = intel3_alpha*(intel3_sigma**-1)*intel3_alpha
microsoft3_stat = microsoft3_alpha*(microsoft3_sigma**-1)*microsoft3_alpha
nvidia3_stat = nvidia3_alpha*(nvidia3_sigma**-1)*nvidia3_alpha
qualcomm3_stat = qualcomm3_alpha*(qualcomm3_sigma**-1)*qualcomm3_alpha
starbucks3_stat = starbucks3_alpha*(starbucks3_sigma**-1)*starbucks3_alpha
asset3_sum = apple3_stat + amazone3_stat + cisco3_stat+ebay3_stat+intel3_stat+microsoft3_stat+nvidia3_stat+qualcomm3_stat+starbucks3_stat
print(asset3_sum)

###############################################################################
####Calculate squared mean and std deviation sigma-squared for nasdaq##########
###############################################################################


mu_nasdaq3 = np.mean(nasdaq3_er)
std_nasdaq3 = np.std(nasdaq3_er)



###############################################################################
#######Calculate Wald test statistic###########################################
###############################################################################


w3 = 48*(1 + mu_nasdaq3**2 / std_nasdaq3)**-1 * asset3_sum
print(w3)


###############################################################################
####Calculate F-statistic J_1##################################################
###############################################################################


J_13 = (48-10-1)/10 *(1 + mu_nasdaq3**2 / std_nasdaq3)**-1 * asset3_sum
print(J_13)


####Calculate the CAPM F test for economic interpretation###



###############################################################################
###############################################################################
##### divide portfolio fourth subperiod 11/2013 - 11/2017   ###################
###############################################################################
###############################################################################



nasdaq4_log_returns = nasdaq_log_returns[147:195]
nasdaq4_er=nasdaq_er[147:195]

amazon4_er=amazon_er[147:195]
apple4_er=apple_er[147:195]
microsoft4_er=microsoft_er[147:195]
cisco4_er=cisco_er[147:195]
comcast4_er=comcast_er[147:195]
ebay4_er=ebay_er[147:195]
intel4_er=intel_er[147:195]
nvidia4_er=nvidia_er[147:195]
qualcomm4_er=qualcomm_er[147:195]
starbucks4_er= starbucks_er[147:195]



######OLS Regression for CAPM alpha and beta###################################
###############################################################################
results_apple4 = smf.ols(formula= 'apple4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_apple4.summary())
results_amazon4 = smf.ols(formula= 'amazon4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_amazon4.summary())
results_microsoft4 = smf.ols(formula= 'microsoft4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_microsoft4.summary())

results_cisco4 = smf.ols(formula= 'cisco4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_cisco4.summary())
results_comcast4 = smf.ols(formula= 'comcast4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_comcast4.summary())
results_ebay4 = smf.ols(formula= 'ebay4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_ebay4.summary())
results_intel4 = smf.ols(formula= 'intel4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_intel4.summary())
results_nvidia4 = smf.ols(formula= 'nvidia4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_nvidia4.summary())
results_qualcomm4 = smf.ols(formula= 'qualcomm4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_qualcomm4.summary())
results_starbucks4 = smf.ols(formula= 'starbucks4_er ~ nasdaq4_er', data=nasdaq4_log_returns).fit()
print(results_starbucks4.summary())

###############################################################################
####Use estimated residuals of residuals for each asset to compute sigma_hat #####
###############################################################################
####apple4##
predictedValues = results_apple4.predict()
res_apple4 = np.array([apple4_er - predictedValues])
res_apple4.shape

apple4_res_transpose=np.transpose(res_apple4)
apple4_sigma=(1/202)*np.matmul(res_apple4 , apple4_res_transpose)
print(apple4_sigma)

####amazon4##
predictedValues_amazon4 = results_amazon4.predict()
res_amazon4 = np.array([amazon4_er - predictedValues_amazon4])

amazon4_res_transpose=np.transpose(res_amazon4)
amazon4_sigma=(1/202)*np.matmul(res_amazon4 , amazon4_res_transpose)
print(amazon4_sigma)

####microsoft4##
predictedValues_microsoft4 = results_microsoft4.predict()
res_microsoft4 = np.array([microsoft4_er - predictedValues_microsoft4])
res_microsoft4.shape

microsoft4_res_transpose=np.transpose(res_microsoft4)
microsoft4_sigma=(1/202)*np.matmul(res_microsoft4 , microsoft4_res_transpose)
print(microsoft4_sigma)

####cisco4##
predictedValues_cisco4 = results_cisco4.predict()
res_cisco4 = np.array([cisco4_er - predictedValues_cisco4])

cisco4_res_transpose=np.transpose(res_cisco4)
cisco4_sigma=(1/202)*np.matmul(res_cisco4 , cisco4_res_transpose)
print(cisco4_sigma)

####comcast4##
predictedValues_comcast4 = results_comcast4.predict()
res_comcast4 = np.array([comcast4_er - predictedValues_comcast4])

comcast4_res_transpose=np.transpose(res_comcast4)
comcast4_sigma=(1/202)*np.matmul(res_comcast4 , comcast4_res_transpose)
print(comcast4_sigma)

####ebay4##
predictedValues_ebay4 = results_ebay4.predict()
res_ebay4 = np.array([ebay4_er - predictedValues_ebay4])

ebay4_res_transpose=np.transpose(res_ebay4)
ebay4_sigma=(1/202)*np.matmul(res_ebay4 , ebay4_res_transpose)
print(ebay4_sigma)

####intel4##
predictedValues_intel4 = results_intel4.predict()
res_intel4 = np.array([intel4_er - predictedValues_intel4])

intel4_res_transpose=np.transpose(res_intel4)
intel4_sigma=(1/202)*np.matmul(res_intel4, intel4_res_transpose)
print(intel4_sigma)

####nvidia4##
predictedValues_nvidia4 = results_nvidia4.predict()
res_nvidia4 = np.array([nvidia4_er - predictedValues_nvidia4])

nvidia4_res_transpose=np.transpose(res_nvidia4)
nvidia4_sigma=(1/202)*np.matmul(res_nvidia4 , nvidia4_res_transpose)
print(nvidia4_sigma)

####qualcomm4##
predictedValues_qualcomm4 = results_qualcomm4.predict()
res_qualcomm4 = np.array([qualcomm4_er - predictedValues_qualcomm4])

qualcomm4_res_transpose=np.transpose(res_qualcomm4)
qualcomm4_sigma=(1/202)*np.matmul(res_qualcomm4 , qualcomm4_res_transpose)
print(qualcomm4_sigma)

####starbucks4##
predictedValues_starbucks4 = results_starbucks4.predict()
res_starbucks4 = np.array([starbucks4_er - predictedValues_starbucks4])

starbucks4_res_transpose=np.transpose(res_starbucks4)
starbucks4_sigma=(1/202)*np.matmul(res_starbucks4 ,starbucks4_res_transpose)
print(starbucks4_sigma)
###############################################################################
####get alpha values for each stock############################################
###############################################################################

apple4_alpha = results_apple4.params['Intercept']
amazon4_alpha = results_amazon4.params['Intercept']
cisco4_alpha = results_cisco4.params['Intercept']
comcast4_alpha = results_comcast4.params['Intercept']
ebay4_alpha = results_ebay4.params['Intercept']
intel4_alpha = results_intel4.params['Intercept']
microsoft4_alpha = results_microsoft4.params['Intercept']
nvidia4_alpha = results_nvidia4.params['Intercept']
qualcomm4_alpha = results_qualcomm4.params['Intercept']
starbucks4_alpha = results_starbucks4.params['Intercept']
###############################################################################
#####Compute (alpha'*Sigma^-1*alpha) as _stat for each asset and sum of all####
###############################################################################

apple4_stat = apple4_alpha*(apple4_sigma**-1)*apple4_alpha
amazon4_stat = amazon4_alpha*(amazon4_sigma**-1)*amazon4_alpha
cisco4_stat = cisco4_alpha*(cisco4_sigma**-1)*cisco4_alpha
comcast4_stat = comcast4_alpha*(comcast4_sigma**-1)*comcast4_alpha
ebay4_stat = ebay4_alpha*(ebay4_sigma**-1)*ebay4_alpha
intel4_stat = intel4_alpha*(intel4_sigma**-1)*intel4_alpha
microsoft4_stat = microsoft4_alpha*(microsoft4_sigma**-1)*microsoft4_alpha
nvidia4_stat = nvidia4_alpha*(nvidia4_sigma**-1)*nvidia4_alpha
qualcomm4_stat = qualcomm4_alpha*(qualcomm4_sigma**-1)*qualcomm4_alpha
starbucks4_stat = starbucks4_alpha*(starbucks4_sigma**-1)*starbucks4_alpha
asset4_sum = apple4_stat + amazon4_stat + cisco4_stat+ebay4_stat+intel4_stat+microsoft4_stat+nvidia4_stat+qualcomm4_stat+starbucks4_stat
print(asset4_sum)



###############################################################################
####Calculate squared mean and std deviation sigma-squared for nasdaq##########
###############################################################################


mu_nasdaq4 = np.mean(nasdaq4_er)
std_nasdaq4 = np.std(nasdaq4_er)



###############################################################################
#######Calculate Wald test statistic###########################################
###############################################################################


w4 = 48*(1 + mu_nasdaq4**2 / std_nasdaq4)**-1 * asset4_sum
print(w4)


###############################################################################
####Calculate F-statistic J_14##################################################
###############################################################################


J_14 = (48-10-1)/10 *(1 + mu_nasdaq4**2 / std_nasdaq4)**-1 * asset4_sum
print(J_14)


######Determining p-Values for Wald test 10 degrees of freedom ######
print(w) #p value 0.20 not significant ?
print(w1)#p value <0.001 not significant
print(w2)#p value <0.001 not significant
print(w3)#p value <0.001 not significant
print(w4)#p value <0.001 not significant


####Determining F test values###
print(J_1)
print(J_11)
print (J_12)
print (J_13)
print (J_14)

###############################################################################
###############################################################################
########                          #############################################
######## CROSS-SECTIONAL ANALYSIS #############################################
########                          #############################################
###############################################################################
###############################################################################


###############################################################################
####regress the excess returns on the stock betas for overall period###########
###############################################################################


###create (Nx1) vector of the estimated market betas for each asset############

apple_betas = [0.9876] * 202
amazon_betas = [0.9951] * 202
microsoft_betas = [0.9944]* 202
cisco_betas = [ 1.0104]* 202
comcast_betas = [1.0076]* 202
ebay_betas = [1.0267]* 202
intel_betas = [ 1.0106]* 202
nvidia_betas = [ 0.9935]* 202
qualcomm_betas = [ 1.0058]* 202
starbucks_betas = [ 0.9965]* 202


###estimate equation Z_t = gamma_0 + gamma_1xbeta##############################

cross_section_apple = smf.ols(formula= 'apple_er ~ apple_betas', data=nasdaq_log_returns).fit()
print(cross_section_apple.summary())

cross_section_amazon = smf.ols(formula= 'amazon_er ~ amazon_betas', data=nasdaq_log_returns).fit()
print(cross_section_amazon.summary())

cross_section_microsoft = smf.ols(formula= 'microsoft_er ~ microsoft_betas', data=nasdaq_log_returns).fit()
print(cross_section_microsoft.summary())

cross_section_cisco = smf.ols(formula= 'cisco_er ~ cisco_betas', data=nasdaq_log_returns).fit()
print(cross_section_cisco.summary())

cross_section_comcast = smf.ols(formula= 'comcast_er ~ comcast_betas', data=nasdaq_log_returns).fit()
print(cross_section_comcast.summary())

cross_section_ebay = smf.ols(formula= 'ebay_er ~ ebay_betas', data=nasdaq_log_returns).fit()
print(cross_section_ebay.summary())

cross_section_intel = smf.ols(formula= 'intel_er ~ intel_betas', data=nasdaq_log_returns).fit()
print(cross_section_intel.summary())

cross_section_nvidia = smf.ols(formula= 'nvidia_er ~ nvidia_betas', data=nasdaq_log_returns).fit()
print(cross_section_nvidia.summary())

cross_section_qualcomm = smf.ols(formula= 'qualcomm_er ~ qualcomm_betas', data=nasdaq_log_returns).fit()
print(cross_section_qualcomm.summary())

cross_section_starbucks = smf.ols(formula= 'starbucks_er ~ starbucks_betas', data=nasdaq_log_returns).fit()
print(cross_section_starbucks.summary())





####use resulting gammas to calculate t-statistics#############################


gamma_0_series = [-0.0074, -0.0147, -0.0030, -0.0040, -0.0014, -0.0006, -0.0031, -0.0042, -0.0015, -0.0043]

t_gamma_0 = np.mean(gamma_0_series)/(np.std(gamma_0_series)/math.sqrt(10))
print(t_gamma_0)

###############################################################################
####regress the excess returns on the stock betas for first subperiod##########
###############################################################################

apple_betas = [0.9520] * 48
amazon_betas = [1.2541] * 48
microsoft_betas = [0.9388]* 48
cisco_betas = [ 1.3314]* 48
comcast_betas = [0.8131]* 48
ebay_betas = [1.2307]* 48
intel_betas = [1.2914]* 48
nvidia_betas = [ 1.5512]* 48
qualcomm_betas = [ 1.1472]* 48
starbucks_betas = [ 0.8050]* 48


###estimate equation Z_t = gamma_0 + gamma_1xbeta##############################

cross_section_apple1 = smf.ols(formula= 'apple1_er ~ apple_betas', data=nasdaq1_log_returns).fit()
print(cross_section_apple1.summary())

cross_section_amazon1 = smf.ols(formula= 'amazon1_er ~ amazon_betas', data=nasdaq1_log_returns).fit()
print(cross_section_amazon1.summary())

cross_section_microsoft1 = smf.ols(formula= 'microsoft1_er ~ microsoft_betas', data=nasdaq1_log_returns).fit()
print(cross_section_microsoft1.summary())

cross_section_cisco1 = smf.ols(formula= 'cisco1_er ~ cisco_betas', data=nasdaq1_log_returns).fit()
print(cross_section_cisco1.summary())

cross_section_comcast1 = smf.ols(formula= 'comcast1_er ~ comcast_betas', data=nasdaq1_log_returns).fit()
print(cross_section_comcast1.summary())

cross_section_ebay1 = smf.ols(formula= 'ebay1_er ~ ebay_betas', data=nasdaq1_log_returns).fit()
print(cross_section_ebay1.summary())

cross_section_intel1 = smf.ols(formula= 'intel1_er ~ intel_betas', data=nasdaq1_log_returns).fit()
print(cross_section_intel1.summary())

cross_section_nvidia1 = smf.ols(formula= 'nvidia1_er ~ nvidia_betas', data=nasdaq1_log_returns).fit()
print(cross_section_nvidia1.summary())

cross_section_qualcomm1 = smf.ols(formula= 'qualcomm1_er ~ qualcomm_betas', data=nasdaq1_log_returns).fit()
print(cross_section_qualcomm1.summary())

cross_section_starbucks1 = smf.ols(formula= 'starbucks1_er ~ starbucks_betas', data=nasdaq1_log_returns).fit()
print(cross_section_starbucks.summary())





####use resulting gammas to calculate t-statistics#############################


gamma_0_series1 = [-0.0107, -0.0131, 0.0075, -0.0010, 0.0015, 0.0023, 0.0002, 0.0059, 0.003, -0.0043]

t1_gamma_0 = np.mean(gamma_0_series1)/(np.std(gamma_0_series1)/math.sqrt(10))
print(t1_gamma_0)



###############################################################################
####regress the excess returns on the stock betas for second subperiod##########
###############################################################################

###create (Nx1) vector of the estimated market betas for each asset############

apple_betas = [0.9720] * 48
amazon_betas = [0.9738] * 48
microsoft_betas = [1.0139]* 48
cisco_betas = [ 0.9935]* 48
comcast_betas = [1.0409]* 48
ebay_betas = [1.0116]* 48
intel_betas = [1.0010]* 48
nvidia_betas = [ 0.9772]* 48
qualcomm_betas = [ 0.9896]* 48
starbucks_betas = [ 0.9786]* 48


###estimate equation Z_t = gamma_0 + gamma_1xbeta##############################

cross_section_apple2 = smf.ols(formula= 'apple2_er ~ apple_betas', data=nasdaq2_log_returns).fit()
print(cross_section_apple2.summary())

cross_section_amazon2 = smf.ols(formula= 'amazon2_er ~ amazon_betas', data=nasdaq2_log_returns).fit()
print(cross_section_amazon2.summary())

cross_section_microsoft2 = smf.ols(formula= 'microsoft2_er ~ microsoft_betas', data=nasdaq2_log_returns).fit()
print(cross_section_microsoft2.summary())

cross_section_cisco2 = smf.ols(formula= 'cisco2_er ~ cisco_betas', data=nasdaq2_log_returns).fit()
print(cross_section_cisco2.summary())

cross_section_comcast2 = smf.ols(formula= 'comcast2_er ~ comcast_betas', data=nasdaq2_log_returns).fit()
print(cross_section_comcast2.summary())

cross_section_ebay2 = smf.ols(formula= 'ebay2_er ~ ebay_betas', data=nasdaq2_log_returns).fit()
print(cross_section_ebay2.summary())

cross_section_intel2 = smf.ols(formula= 'intel2_er ~ intel_betas', data=nasdaq2_log_returns).fit()
print(cross_section_intel2.summary())

cross_section_nvidia2 = smf.ols(formula= 'nvidia2_er ~ nvidia_betas', data=nasdaq2_log_returns).fit()
print(cross_section_nvidia2.summary())

cross_section_qualcomm2 = smf.ols(formula= 'qualcomm2_er ~ qualcomm_betas', data=nasdaq2_log_returns).fit()
print(cross_section_qualcomm2.summary())

cross_section_starbucks2 = smf.ols(formula= 'starbucks2_er ~ starbucks_betas', data=nasdaq2_log_returns).fit()
print(cross_section_starbucks2.summary())





####use resulting gammas to calculate t-statistics#############################


gamma0_series2 = [-0.0557, -0.0501, -0.0407, -0.0444, -0.0341, -0.0350, -0.0388, -0.0334, -0.0417, -0.0327]

t2_gamma_0 = np.mean(gamma0_series2)/(np.std(gamma0_series2)/math.sqrt(10))
print(t2_gamma_0)


###############################################################################
####regress the excess returns on the stock betas for third subperiod##########
###############################################################################

###create (Nx1) vector of the estimated market betas for each asset############
apple_betas = [0.9836] * 48
amazon_betas = [0.9836] * 48
microsoft_betas = [0.9836]* 48
cisco_betas = [ 1.0033]* 48
comcast_betas = [0.9948]* 48
ebay_betas = [1.0007]* 48
intel_betas = [1.0007]* 48
nvidia_betas = [ 1.0016]* 48
qualcomm_betas = [ 1.0016]* 48
starbucks_betas = [1.0168]* 48


###estimate equation Z_t = gamma_0 + gamma_1xbeta##############################
cross_section_apple3 = smf.ols(formula= 'apple3_er ~ apple_betas', data=nasdaq3_log_returns).fit()
print(cross_section_apple3.summary())

cross_section_amazon3 = smf.ols(formula= 'amazone3_er ~ amazon_betas', data=nasdaq3_log_returns).fit()
print(cross_section_amazon3.summary())

cross_section_microsoft3 = smf.ols(formula= 'microsoft3_er ~ microsoft_betas', data=nasdaq3_log_returns).fit()
print(cross_section_microsoft3.summary())

cross_section_cisco3 = smf.ols(formula= 'cisco3_er ~ cisco_betas', data=nasdaq3_log_returns).fit()
print(cross_section_cisco3.summary())

cross_section_comcast3 = smf.ols(formula= 'comcast3_er ~ comcast_betas', data=nasdaq3_log_returns).fit()
print(cross_section_comcast3.summary())

cross_section_ebay3 = smf.ols(formula= 'ebay3_er ~ ebay_betas', data=nasdaq3_log_returns).fit()
print(cross_section_ebay3.summary())

cross_section_intel3 = smf.ols(formula= 'intel3_er ~ intel_betas', data=nasdaq3_log_returns).fit()
print(cross_section_intel3.summary())

cross_section_nvidia3 = smf.ols(formula= 'nvidia3_er ~ nvidia_betas', data=nasdaq3_log_returns).fit()
print(cross_section_nvidia3.summary())

cross_section_qualcomm3 = smf.ols(formula= 'qualcomm3_er ~ qualcomm_betas', data=nasdaq3_log_returns).fit()
print(cross_section_qualcomm3.summary())

cross_section_starbucks3 = smf.ols(formula= 'starbucks3_er ~ starbucks_betas', data=nasdaq3_log_returns).fit()
print(cross_section_starbucks3.summary())





####use resulting gammas to calculate t-statistics#############################


gamma_0_series3 = [-0.0007, -0.0018, 0.0073, 0.0101, -0.0024, 0.0009, 0.0073, 0.0076, 0.0045, -0.0053]

t3_gamma_0 = np.mean(gamma_0_series3)/(np.std(gamma_0_series3)/math.sqrt(10))
print(t3_gamma_0)




###############################################################################
####regress the excess returns on the stock betas for fourth subperiod##########
###############################################################################

###create (Nx1) vector of the estimated market betas for each asset############
apple_betas = [1.0168] * 48
amazon_betas = [1.0168] * 48
microsoft_betas = [0.9899]* 48
cisco_betas = [ 0.9899]* 48
comcast_betas = [0.9899]* 48
ebay_betas = [0.9899]* 48
intel_betas = [0.9899]* 48
nvidia_betas = [0.9899]* 48
qualcomm_betas = [ 0.9899]* 48
starbucks_betas = [ 0.9899]* 48


###estimate equation Z_t = gamma_0 + gamma_1xbeta##############################
cross_section_apple4 = smf.ols(formula= 'apple4_er ~ apple_betas', data=nasdaq4_log_returns).fit()
print(cross_section_apple4.summary())

cross_section_amazon4 = smf.ols(formula= 'amazon4_er ~ amazon_betas', data=nasdaq4_log_returns).fit()
print(cross_section_amazon4.summary())

cross_section_microsoft4 = smf.ols(formula= 'microsoft4_er ~ microsoft_betas', data=nasdaq4_log_returns).fit()
print(cross_section_microsoft4.summary())

cross_section_cisco4 = smf.ols(formula= 'cisco4_er ~ cisco_betas', data=nasdaq4_log_returns).fit()
print(cross_section_cisco4.summary())

cross_section_comcast4 = smf.ols(formula= 'comcast4_er ~ comcast_betas', data=nasdaq4_log_returns).fit()
print(cross_section_comcast4.summary())

cross_section_ebay4 = smf.ols(formula= 'ebay4_er ~ ebay_betas', data=nasdaq4_log_returns).fit()
print(cross_section_ebay4.summary())

cross_section_intel4 = smf.ols(formula= 'intel4_er ~ intel_betas', data=nasdaq4_log_returns).fit()
print(cross_section_intel4.summary())

cross_section_nvidia4 = smf.ols(formula= 'nvidia4_er ~ nvidia_betas', data=nasdaq4_log_returns).fit()
print(cross_section_nvidia4.summary())

cross_section_qualcomm4 = smf.ols(formula= 'qualcomm4_er ~ qualcomm_betas', data=nasdaq4_log_returns).fit()
print(cross_section_qualcomm4.summary())

cross_section_starbucks4 = smf.ols(formula= 'starbucks4_er ~ starbucks_betas', data=nasdaq4_log_returns).fit()
print(cross_section_starbucks4.summary())


####use resulting gammas to calculate t-statistics#############################


gamma_0_series4 = [0.0463, 0.0219, 0.0242, 0.0271, 0.0343, 0.0363, 0.0262, 0.0063, 0.0342, 0.0371]

t4_gamma_0 = np.mean(gamma_0_series4)/(np.std(gamma_0_series4)/math.sqrt(10))
print(t4_gamma_0)

