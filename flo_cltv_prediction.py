# DATA UNDERSTANDING AND PREPARATION                 

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
# pd.set_option('display_max_rows',None)
pd.set_option('display.float_format',lambda x : '%.5f' % x)

# READ DATASET
df_= pd.read_csv("flo_data_20k.csv")
df = df_.copy()

# CHECKING THE DATA

def datacheck(dataframe):
    print("******Head******")
    print(dataframe.head(10))
    print("******Shape******")
    print(dataframe.shape)
    print("******Info********")
    print(dataframe.info())
    print("******Describe********")
    print(dataframe.describe().T)
    print("***** NAN Values********")
    print(dataframe.isnull().sum())

datacheck(df)

# Define the function for outlier threshold and replace with threshold values

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T

df.describe(percentiles=[.50,.60,.70,.80,.90]).T

# Total purchases for omnichannel customers
df["total_purchases_number"] = df["order_num_total_ever_online"] +  df["order_num_total_ever_offline"]

# Total expense for omnichannel customers
df["total_customer_expense"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)
df.info()

# CREATING CLTV DATA STRUCTURE                          

df["last_order_date"].max()

last_date = dt.datetime(2021,5,30)
type(last_date)

today_date = dt.datetime(2021, 6, 2)
type(today_date)

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((today_date - df["first_order_date"]).dt.days)/7,
             "frequency": df["total_purchases_number"],
             "monetary_cltv_avg": df["total_customer_expense"] / df["total_purchases_number"]})


cltv_df.head()

#CREATING BG/NBD,GAMMA GAMMA MODELS AND CALCULATING CLTV            

#Fit the BG/NBD Model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
       cltv_df['recency_cltv_weekly'],
       cltv_df['T_weekly'])

bgf.summary

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)
plt.show(block=True)

from lifetimes.plotting import plot_probability_alive_matrix
fig = plt.figure(figsize=(12,8))
plot_probability_alive_matrix(bgf);

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])

cltv_df.sort_values(by='exp_sales_3_month', ascending=False).head()

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])

# Alternative

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df['probability_alive'] = bgf.conditional_probability_alive(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])
cltv_df.head(10)

cltv_df.head(20)

# Bonus

bgf.conditional_expected_number_of_purchases_up_to_time(
    4*3, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"]
).sort_values(ascending=False).head(10)

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df['frequency'],
                                                        cltv_df['recency_cltv_weekly'],
                                                        cltv_df['T_weekly'])

cltv1= cltv_df.drop(cltv_df.loc[:,'recency_cltv_weekly':'monetary_cltv_avg'].columns, axis=1)

cltv1.sort_values(by='exp_sales_3_month', ascending=False).head()

plot_period_transactions(bgf, max_frequency=7)
plt.show(block=True)

from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)
plt.show(block=True)

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)
plt.show(block=True)

# Fit the Gamma Gamma Model

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

ggf.summary

cltv_df["exp_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])

cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,  # 6 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)


cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=12,  # 12 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:10]

# CREATING SEGMENT ACCORDING CLTV VALUES                             

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D","C", "B", "A"])

cltv_df.head()

stats=["mean","sum","count"]

cltv_df.groupby("cltv_segment").agg({"exp_sales_3_month":stats,"exp_sales_6_month":stats,"exp_average_profit":stats})

# Write a csv file to a new folder

from pathlib import Path
filepath = Path('D:/12thTerm_DS_Bootcamp/3Week_CRM_Analytics/cltv.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
cltv_df.to_csv(filepath)

# BONUS

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 3, labels=["C", "B", "A"])
cltv_df.groupby("cltv_segment").agg({"count"})

cltv_df.drop("customer_id",axis=1,inplace=True)

cltv_seg=cltv_df.loc[:,"monetary_cltv_avg":"cltv_segment"]

cltv_df.groupby("cltv_segment").agg({"mean","sum"})
