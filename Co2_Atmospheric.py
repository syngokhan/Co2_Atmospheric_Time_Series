#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing , ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import graphics

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm


# In[3]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[4]:


pd.set_option("display.max_columns",None)
pd.set_option("display.float_format", lambda x : "%.4f" %x)
pd.set_option("display.width", 200)


# In[5]:


############################
# Data set
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001


# In[6]:


data = sm.datasets.co2.load_pandas()
type(data)


# In[7]:


y = data.data
type(y)


# In[8]:


# We can review it from here.....

y.head(20).T


# In[9]:


y.index.sort_values()


# In[10]:


# Here we took the average of what happened in the same month...

y.loc["1958-04-05":"1958-04-26"].mean()


# In[11]:


y.loc["1958-05-02":"1958-05-24"].mean()


# In[12]:


y["co2"].resample("MS").mean()


# In[13]:


y = y[["co2"]].resample("MS").mean()
type(y)


# In[14]:


# It already has missing values itself...

y.isnull().sum()


# In[15]:


# It fills itself with the next ...

y["co2_Na_Values"] = y.fillna(y.backfill())
y.head(20).T


# In[16]:


plt.figure(figsize = (15,10))

plt.subplot(2,1,1)
y["co2"].plot( color = "b",label = "There Are Missing Values")
y["co2_Na_Values"].plot(color = "g", label = "There Aren't Missing Values")
plt.legend(loc = "best")
plt.title("Compare Na_Values vsv Orginal Data",fontsize = 15)

plt.subplot(2,1,2)
y["co2"].plot( color = "b",label = "There Are Missing Values")
y["co2_Na_Values"].plot(color = "g", label = "There Aren't Missing Values")
plt.xlim(["1963","1968"])

plt.legend(loc="best")
plt.show()


# In[17]:


# We Take Series Attention !!!

import statsmodels.api as sm

y = sm.datasets.co2.load_pandas().data
y = y["co2"].resample("MS").mean()
y = y.fillna(y.bfill())
type(y)


# In[18]:


# There Aren't Na Values...

y.isnull().sum()


# In[19]:


pd.DataFrame(y).head(20).T


# In[20]:


y.plot(figsize = (15,6), label = "Orjinal Data")
plt.title("Orginal Data", fontsize = 15)
plt.legend(loc = "best")
plt.show()


# In[21]:


############################
# Holdout
############################


# In[22]:


# Train set from 1958 to the end of 1997
# We want to predict 48 months ahead...


# In[23]:


train_number = len(y)-48
train_number


# In[24]:


train = y.iloc[:train_number]
test = y.iloc[train_number:]

print("Train Shape : {}".format(train.shape))
print("Test Shape : {}".format(test.shape))


# In[25]:


# For Train 

print("Min Date Train : {}".format(train.index.min()))
print("Max Date Train : {}".format(train.index.max()))


# In[26]:


# For Test

print("Min Date Train : {}".format(test.index.min()))
print("Max Date Train : {}".format(test.index.max()))


# In[27]:


################################################
# Time Series Structural Analysis
################################################


# In[28]:


# Stability Test (Dickey - Fuller Test)


# In[29]:


from statsmodels.tsa.api import adfuller


# In[30]:


def is_stationary(model , plot = False):
    
    print("""
    *************************
    H0 : Non-Stationary
    H1 : Stationary
    *************************
    """)
    
    from statsmodels.tsa.api import adfuller
    
    p_value = adfuller(model)[1]
    
    if p_value < 0.05 :
        
        print(f"Results : Stationary (H0: Non-Stationary , p-value : {round(p_value,4)})")
        
    else:
        
        print(f"Results : Non-Stationary (H0 : Non-Stationary , p-value : {round(p_value,4)})")
        
    
    if plot :
        
        plt.figure(figsize = (15,6))
        plt.plot(y)
        plt.title(f"Graph For CO2 (P-Value : {round(p_value,3)})", fontsize = 15)
        plt.show()


# In[31]:


# P_Values < 0.05 Red
# P_Values < 0.05 değilse Red edilemez...

is_stationary(y,plot = True)


# In[32]:


def ts_decompose(y, model = "additive",stationary = False):
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    result = seasonal_decompose(y , model = model)
    
    fig , axes = plt.subplots(4,1, sharex= True ,sharey = False)
    
    fig.set_figheight(10)
    fig.set_figwidth(15)
    
    axes[0].set_title("Decompose For " + model.upper() + " Model", fontsize = 15)
    axes[0].plot(y, "k", label = "Orginal "+ model.upper())
    axes[0].legend(loc = "upper left")
    
    axes[1].plot(result.trend, "r", label = "Trend")
    axes[1].legend(loc = "upper left")
    
    axes[2].plot(result.seasonal, "g", label = "Seasonility & Mean : " + str(round(result.seasonal.mean(),4)))
    axes[2].legend(loc = "upper left")
    
    axes[3].plot(result.resid , "b", label = "Residuals & Mean : " + str(round(result.resid.mean(),4 )))
    axes[3].legend(loc = "upper left")
    
    
    if stationary:
        
        is_stationary(y, plot = False)


# In[33]:


ts_decompose(y,model = "additive", stationary=True)


# In[34]:


# Analysis for additive and multiplicative models

# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise

for model in ["additive","multiplicative"]:
    ts_decompose(y, model,stationary=False)


# In[35]:


################################################
# Single Exponential Smoothing
################################################

# SOUND = Level
# Used in stationary series.
# Cannot be used if there are trends and seasonality.


# In[36]:


# We need to check simple model 

def plot_co2_graph(train,test,y_pred,title):
    
    mae = mean_absolute_error(test,y_pred)
    title = title + f" (MAE : {round(mae, 4)})"
    plt.figure(figsize = (15,7))
    
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    y_pred.plot(label = "PREDICTION")
    
    plt.legend(loc = "upper left")
    plt.title(title , fontsize = 15)
    plt.show()


# In[37]:


from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error


# In[38]:


ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
ses_y_pred = ses_model.forecast(steps = 48)

plot_co2_graph(train,test,ses_y_pred, title = "Base Simple Exponential Smoothing ")


# In[39]:


########################################################
# Hyperparameter Optimization & FINAL SES Model
########################################################


# In[40]:


def ses_optimizer(train,test,alphas,steps = 48):
    
    best_alpha, best_mae = None, float("inf")
    
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level = alpha)
        ses_y_pred = ses_model.forecast(steps = steps)
        ses_mae = mean_absolute_error(test, ses_y_pred)
        
        if ses_mae < best_mae:
            
            best_alpha, best_mae = alpha,ses_mae
            
        print(f"Alpha : {round(alpha,4)}, MAE : {round(ses_mae,4)}")
        
    print("\n\n")
    print(f"Best Alpha : {round(best_alpha,4)}, Best MAE : {round(best_mae,4)}")
    
    return best_alpha


# In[41]:


alphas = np.arange(0.8 ,1 ,0.01)

ses_best_alpha = ses_optimizer(train , test , alphas , steps = 48)


# In[42]:


def final_ses_graph(train, test, best_alpha, steps = 48):
    
    final_ses_model = SimpleExpSmoothing(train).fit(smoothing_level = best_alpha)
    final_ses_y_pred = final_ses_model.forecast(steps)
    final_ses_mae = mean_absolute_error(test, final_ses_y_pred)
    
    title = f"Simple Exponential Smoothing (MAE : {round(final_ses_mae , 4)} , Alpha : {round(best_alpha,4)})"
    title = "Optimizer " + title
    
    plt.figure(figsize = (15,7))
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    final_ses_y_pred.plot(label = "PREDICTION")
    
    plt.title(title , fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[43]:


plot_co2_graph(train,test,ses_y_pred,title = "Base Simple Exponential Smoothing ")
final_ses_graph(train, test, ses_best_alpha, steps = 48)


# In[44]:


################################################
# Double Exponential Smoothing (DES)
################################################

# DES: Level (SES) + Trend

# In addition to Level, it can catch the Trend.


# In[45]:


des_model = ExponentialSmoothing(train,trend = "add").fit(smoothing_level = 0.5,                                                           smoothing_trend = 0.5)

des_y_pred = des_model.forecast(steps = 48)

plot_co2_graph(train, test , des_y_pred, "Double Exponential Smoothing ")


# In[46]:


########################################################
# Hyperparameter Optimization & Des Model
########################################################


# In[47]:


def des_optimizer(train, test, alphas, betas, trend = "add", steps = 48):
    
    best_alpha, best_beta ,best_mae = None, None, float("inf")
    
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train,trend = "add").fit(smoothing_level = alpha,
                                                                      smoothing_trend = beta)
            des_y_pred = des_model.forecast(steps = steps)
            des_mae = mean_absolute_error(test, des_y_pred)
            
            if des_mae < best_mae:
                
                best_alpha,best_beta,best_mae = alpha,beta,des_mae
                
            print(f"Alpha : {round(alpha,4)}, Beta : {round(beta,4)}, MAE : {round(des_mae , 4)}")
            
    
    print("\n\n")
    print(f"Best Alpha : {round(best_alpha,4)}, Best Beta : {round(best_beta,4)},Best MAE : {round(best_mae,4)}")
    
    return best_alpha, best_beta


# In[48]:


alphas = np.arange(0.01,1,0.1)
betas = np.arange(0.01,1,0.1)

des_best_alpha,des_best_beta = des_optimizer(train, test, alphas,betas,trend = "add", steps =48)


# In[49]:


def final_des_graph(train,test,best_alpha,best_beta,steps=48):
    
    final_des_model = ExponentialSmoothing(train,trend = "add").fit(smoothing_level = best_alpha,                                                                    smoothing_trend = best_beta)
    final_des_y_pred= final_des_model.forecast(steps = steps)
    
    final_des_mae = mean_absolute_error(test, final_des_y_pred)
    
    title = "Double Exponential Smoothing"
    title = title + f" (MAE : {round(final_des_mae,4)}, Alpha : {round(best_alpha,4)}, Beta : {round(best_beta,4)})"
    title = "Optimizer " + title
    
    plt.figure(figsize = (15,7))
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    final_des_y_pred.plot(label = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[50]:


plot_co2_graph(train,test,des_y_pred,title = "Base Double Exponential Smoothing ")
final_des_graph(train,test,des_best_alpha,des_best_beta,steps = 48)


# In[51]:


################################################
# Triple Exponential Smoothing (Holt-Winters)
################################################


# TES = SES + DES + Seasonality


# In[52]:


tes_model = ExponentialSmoothing(train,
                                  trend = "add",
                                  seasonal="add",
                                  seasonal_periods=12).fit(smoothing_level=0.5,
                                                           smoothing_trend=0.5,
                                                           smoothing_seasonal=0.5)

tes_y_pred = tes_model.forecast(steps = 48)

plot_co2_graph(train, test, tes_y_pred, title = "Triple Exponential Smoothing ")


# In[53]:


########################################################
# Hyperparameter Optimization & Tes Model
########################################################


# In[54]:


def tes_optimizer(train, test, abg ,trend = "add",seasonal = "add",seasonal_periods = 12, steps =48):
    
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    
    for comb in abg:
        
        tes_model = ExponentialSmoothing(train,
                                          trend=trend,
                                          seasonal=seasonal,
                                          seasonal_periods=seasonal_periods).fit(smoothing_level = comb[0],
                                                                                 smoothing_trend = comb[1],
                                                                                 smoothing_seasonal = comb[2])
        tes_y_pred = tes_model.forecast(steps = steps)
        
        tes_mae = mean_absolute_error(test, tes_y_pred)
        
        if tes_mae < best_mae:
            
            best_alpha,best_beta,best_gamma,best_mae = comb[0],comb[1],comb[2],tes_mae
        
        
        
        print(f"Alpha : {round(comb[0],3)},Beta : {round(comb[1],3)},Gamma : {round(comb[2],3)},MAE : {round(tes_mae,3)}")
    
    print("\n\n")
    print(f"Best Alpha : {round(best_alpha,3)},Best Beta : {round(best_beta,3)},Best Gamma : {round(best_gamma,3)},Best MAE : {round(best_mae,3)},")
    
    return best_alpha,best_beta,best_gamma


# In[55]:


import itertools

alphas = betas = gammas = np.arange(0.1, 1, 0.2)
abg = list(itertools.product(alphas, betas, gammas))

tes_best_alpha,tes_best_beta,tes_best_gamma = tes_optimizer(train,
                                                            test,
                                                            abg,
                                                            trend = "add",
                                                            seasonal="add",
                                                            seasonal_periods=12,
                                                            steps = 48)


# In[56]:


def final_tes_graph(train,test,best_alpha,best_beta,best_gamma,trend="add",
                    seasonal="add",seasonal_periods = 12,steps = 48):
    
    final_tes_model = ExponentialSmoothing(train,
                                           trend=trend,
                                           seasonal=seasonal,
                                           seasonal_periods=seasonal_periods).fit(smoothing_level=best_alpha,
                                                                                  smoothing_trend=best_beta,
                                                                                  smoothing_seasonal=best_gamma)
    
    final_tes_y_pred = final_tes_model.forecast(steps = steps )
    
    final_tes_mae = mean_absolute_error(test, final_tes_y_pred)
    
    title = "Triple Exponential Smoothing "
    title = title + f"(MAE : {round(final_tes_mae,4)},Alpha : {round(best_alpha,4)},Beta : {round(best_beta,4)},Gamma : {round(best_gamma,4)})"
    title = "Optimizer " + title
    
    plt.figure(figsize = (15,7))
    train["1985": ].plot(label = "TRAIN")
    test.plot(label = "TEST")
    final_tes_y_pred.plot(legend = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[57]:


plot_co2_graph(train,test,tes_y_pred,title = "Base Triple Exponential Smoothing ")
final_tes_graph(train,test,tes_best_alpha,tes_best_beta,tes_best_gamma)


# ---

# ## Comparison Base Vs Optimizer

# In[58]:


plot_co2_graph(train, test, ses_y_pred, "Base Simple Exponential Smoothing ")
final_ses_graph(train,test, ses_best_alpha)


# In[59]:


plot_co2_graph(train, test, des_y_pred, "Base Double Exponential Smoothing ")
final_des_graph(train,test, des_best_alpha,des_best_beta)


# In[60]:


plot_co2_graph(train, test, tes_y_pred, "Base Triple Exponential Smoothing ")
final_tes_graph(train,test, tes_best_alpha, tes_best_beta,tes_best_gamma)


# ---

# In[61]:


##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################


# In[62]:


from statsmodels.tsa.arima_model import ARIMA


# In[63]:


# p : Geçmiş Gerçek Değer Geçikme Sayısı
# q : Geçmiş Artık hatalardaki geçikme sayısı
# d : Fark İşlemi Sayısı (Fark Derecesi ,1)

# p : Past Actual Value Latency Count
# q : History Number of delays in residual errors
# d : Number of Difference Operations (Differential Degree ,1)

# disp = verbose

arima_model = ARIMA(train ,order = (1,1,1)).fit(disp = 0)
arima_model.summary()


# In[64]:


# Burda indexler belirlenmiyor bizim eklememiz lazım 
# Arıma sadece trend ve level yakalayabiliyor...

# Indexes are not determined here, we need to add
# Arima can only catch trend and level...

arima_y_pred = arima_model.forecast(steps = 48)[0]
arima_y_pred = pd.Series(data = arima_y_pred, index = test.index)
plot_co2_graph(train, test, arima_y_pred, "Base ARIMA")


# In[65]:


# Buda kendi özelliği

# Here is its own feature

fig,axes = plt.subplots(1,1,figsize =(15,7))

arima_model.plot_predict(ax = axes, dynamic = False)
plt.title("ARIMA PREDICTIONS FEATURES ",fontsize = 15)
plt.show()


# In[66]:


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################

# 1. AIC İstatistiğine Göre Model Derecesini Belirleme
# 2. ACF & PACF Grafiklerine Göre Model Derecesini Belirleme

############################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme
############################


############################
# Hyperparameter Optimization (Determining Model Grades)
############################

#1. Determining Model Grade Based on AIC Statistics
#2. Determining Model Grade Based on ACF & PACF Charts

############################
# Determining Model Grade Based on AIC & BIC Statistics
############################


# Akaike ölçütü (Akaike information criterion-AIC) belirli bir veri kümesi için kaliteli 
# bir istatistiksel göreceli model ölçüsüdür. 
# Yani, veri modelleri koleksiyonu verildiğinde, AIC her model kalitesini, diğer modellerin her birini göreceli olarak tahmin ediyor.
# Dolayısıyla, AIC model seçimi için bir yol sağlar.

# Akaike criterion (Akaike information criterion-AIC) for a given dataset with good quality
# is a statistical relative model measure.
# That is, given a collection of data models, AIC predicts each model quality relative to each of the other models.
# Hence, AIC provides a way for model selection.


# In[67]:


p = d = q =range(0,4)
pdq = list(itertools.product(p,d,q))

def arima_optimizer_aic(train,orders):
    
    best_aic, best_params = float("inf"), None
    
    for order in orders:
        try:
            arima_model_results =ARIMA(train, order = order).fit(disp = 0)
            aic = arima_model_results.aic
            if aic < best_aic:
                best_aic, best_params = aic , order
        
            print(f"AIC : {round(aic,4)} , Order : {order}")
        
        except :
            continue 
    
    print("\n\n")
    print(f"Best AIC : {round(best_aic , 4)} ,Best Order : {best_params}")
    
    return best_params


# In[68]:


def arima_optimizer_mae(train,test,orders):
    
    best_mae, best_params = float("inf"), None
    
    for order in orders:
       
        try:
            arima_model = ARIMA(train, order = order).fit(disp = 0)
            arima_y_pred = arima_model.forecast(steps = 48)[0]
            mae = mean_absolute_error(test, arima_y_pred)
        
            if mae < best_mae:
            
                best_mae , best_params = mae, order
            
            
            print(f"MAE : {round(mae, 4)}, Order : {order}")
        
        except:
            continue
            
    print("\n\n")
    print(f"Best MAE : {round(best_mae,4)}, Best Order : {best_params}")
    
    return best_params


# In[69]:


arima_best_params_mae = arima_optimizer_mae(train, test, pdq)


# In[70]:


arima_best_params_aic = arima_optimizer_aic(train, pdq)


# In[71]:


def final_arima_aic(train, test, best_params, steps = 48 ):
    
    final_arima_model_aic = ARIMA(train, order = best_params).fit(disp = 0)
    final_y_pred_aic = final_arima_model_aic.forecast(steps = steps )[0]
    final_aic = final_arima_model_aic.aic
    predictions = pd.Series(data = final_y_pred_aic, index = test.index)
    
    
    title = f"Optimizer ARIMA Model (AIC : {round(final_aic, 4)}, Order : {best_params} )"
    plt.figure(figsize = (15,7))
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    predictions.plot(label = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[72]:


def final_arima_mae(train,test,best_params, steps = 48):
    
    final_arima_model_mae = ARIMA(train , order = best_params).fit(disp = 0)
    final_y_pred_mae = final_arima_model_mae.forecast(steps = steps)[0]
    final_mae = mean_absolute_error(test, final_y_pred_mae)
    predictions = pd.Series(data = final_y_pred_mae, index = test.index)
    
    title = f"Optimizer ARIMA Model (MAE : {round(final_mae, 4)}, Order : {best_params} )"
    plt.figure(figsize = (15,7))
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    predictions.plot(label = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[73]:


final_arima_mae(train,test,arima_best_params_mae,steps = 48)
final_arima_aic(train,test,arima_best_params_aic,steps = 48)


# In[74]:


############################
# ACF & PACF Grafiklerine Göre Model Derecesini Belirleme
############################

############################
# Determining Model Grade According to ACF & PACF Charts
############################


# In[75]:


def acf_pacf(model, lags = 30):
    
    plt.figure(figsize = (15,7))
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1,0))
    pacf_ax = plt.subplot2grid(layout, (1,1))
    model.plot(ax = ts_ax)
    
    from statsmodels.tsa.api import adfuller
    
    
    p_value = adfuller(model)[1]
    ts_ax.set_title("Time Series Analysis Plots\n Dickey-Fuller : p = {0:.5f}".format(p_value))
    
    from statsmodels.tsa.api import graphics
    
    graphics.plot_pacf(model, lags = lags, ax = pacf_ax)
    graphics.plot_acf(model, lags = lags , ax = acf_ax)
    plt.tight_layout()
    plt.show()


# In[76]:


# Bir zaman serisindeki önceki periodların birbirleriyle olan korelasyonlarıdır !!!!
# Çizgilerin içinde kalmaması lazım..(Yuvarlaklar hariç) İstatiki olarak anlamlı değildir !!!!
# P ve Q karar veriyoruz !!!!

# ACF genişliği gecikmelere göre "AZALIYORSA" ve PACF p gecikme sonra "KESILIYORSA" AR(p) modeli olduğu anlamına gelir.

# ACF genişliği q gecikme sonra "KESILIYORSA" ve PACF genişliği gecikmelere göre "AZALIYORSA" MA(q) modeli olduğu anlamına gelir.

# ACF ve PACF'nin genişlikleri gecikmelere göre azalıyorsa, ARMA modeli olduğu anlamına gelir.

# Correlations of previous periods in a time series with each other !!!!
# It should not stay inside the lines. (Except for the circles) It is not statistically significant !!!!
# We decide P and Q !!!!

# IF ACF width "DECREASE" relative to the delays and PACF "CUT" after the p delay means it's an AR(p) pattern.

# If the ACF width q "CUT" after the delay and the PACF width "DECREASE" according to the delays, it means it's a MA(q) pattern.

# If the widths of ACF and PACF are decreasing with respect to the delays, it means it is an ARMA model.

acf_pacf(y)


# In[77]:


# Yukarıdakiden anlamlı bir çıkarım yapamadım ...

# I couldn't make any meaningful inferences from the above...

acf_pacf(y.diff(1).dropna())


# In[78]:


print("Best AIC Order Params : {}".format(arima_best_params_mae))
print("Best MAE Order Params : {}".format(arima_best_params_aic))


# In[79]:


##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################


# In[80]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(train, order = (1,0,1), seasonal_order = (0,0,0,12)).fit(disp = 0)

sarima_y_pred = sarima_model.forecast(48)

plot_co2_graph(train, test, sarima_y_pred, "Base SARIMAX ")


# In[81]:


############################
# Hyperparameter Optimization (Determining Model Grades)
############################


# In[82]:


p = d = q =range(0,2)
pdq = list(itertools.product(p,d,q))
seaosanal_pdq = [(x[0],x[1], x[2], 12) for x in list(itertools.product(p,d,q))]


# In[83]:


def sarima_optimizer_aic(train,pdq,seasonal_pdq):
    
    best_aic, best_order ,best_seasonal_order = float("inf"), None , None
    
    for param in pdq:
        for seasonal_param in seasonal_pdq:
             
                try:
                    sarima_model = SARIMAX(train , order=param, seasonal_order=seasonal_param).fit(disp = 0)
                    sarima_aic = sarima_model.aic
                    
                    if sarima_aic < best_aic :
                        
                        best_aic, best_order, best_seasonal_order = sarima_aic,param,seasonal_param
                    
                    print(f"AIC : {round(sarima_aic,4)},Order : {param},Seasonal Order : {seasonal_param}")
                
                except: 
                    continue
    
    print("\n\n")
    print(f"Best AIC : {round(best_aic,4)},Best Order : {best_order}, Best Seasonal Order : {best_seasonal_order}")
    
    return best_order, best_seasonal_order


# In[84]:


def sarima_optimizer_mae(train, test, pdq,seasonal_pdq,steps = 48):
    
    best_mae , best_order, best_seasonal_order = float("inf"), None,None
    
    for param in pdq:
        for seasonal_param in seaosanal_pdq:
            
            try:
                sarima_model = SARIMAX(train, order = param , seasonal_order=seasonal_param).fit(disp = 0)
                sarima_y_pred = sarima_model.forecast(steps = steps)
                sarima_mae = mean_absolute_error(test, sarima_y_pred)
                
                if sarima_mae < best_mae:
                    
                    best_mae, best_order ,best_seasonal_order = sarima_mae, param, seasonal_param
                    
                print(f"MAE : {round(sarima_mae, 4)}, Order : {param}, Seasonal Order : {seasonal_param}")
                
            except:
                continue
                
    print("\n\n")
    print(f"Best MAE : {round(best_mae,4)}, Best Order : {best_order}, Seasonal Order : {best_seasonal_order}")
    
    return best_order, best_seasonal_order


# In[85]:


best_mae_order_sarimax, best_mae_seasonal_order_sarimax = sarima_optimizer_mae(train, test, pdq, seaosanal_pdq,steps = 48)


# In[86]:


best_aic_order_sarimax, best_aic_seasonal_order_sarimax = sarima_optimizer_aic(train,pdq,seaosanal_pdq)


# In[87]:


########################################################
# Final Model For SARIMAX MAE & AIC
########################################################


# In[88]:


def final_sarimax_mae(train,test,best_order,best_seasonal_order, steps = 48):

    final_sarimax_mae = SARIMAX(train, order = best_order, seasonal_order = best_seasonal_order).fit(disp = 0)
    final_y_sarimax_pred = final_sarimax_mae.forecast(steps = steps )
    final_mae = mean_absolute_error(test, final_y_sarimax_pred)
    
    title = f"Optimizer SARIMAX (MAE : {round(final_mae, 4)}, Order : {best_order}, Seasonal Order : {best_seasonal_order})"
    
    plt.figure(figsize = (15,7))
    
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    final_y_sarimax_pred.plot(label = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[89]:


def final_sarimax_aic(train,test,best_order,best_seasonal_order, steps = 48):

    final_sarimax_aic = SARIMAX(train, order = best_order, seasonal_order = best_seasonal_order).fit(disp = 0)
    final_y_sarimax_pred = final_sarimax_aic.forecast(steps = steps )
    final_aic = final_sarimax_aic.aic
    
    title = f"Optimizer SARIMAX (AIC : {round(final_aic, 4)}, Order : {best_order}, Seasonal Order : {best_seasonal_order})"
    
    plt.figure(figsize = (15,7))
    
    train["1985":].plot(label = "TRAIN")
    test.plot(label = "TEST")
    final_y_sarimax_pred.plot(label = "PREDICTION")
    
    plt.title(title, fontsize = 15)
    plt.legend(loc = "upper left")
    plt.show()


# In[90]:


final_sarimax_aic(train, test, best_aic_order_sarimax, best_aic_seasonal_order_sarimax, steps = 48)
final_sarimax_mae(train, test, best_mae_order_sarimax,best_mae_seasonal_order_sarimax, steps = 48)
final_tes_graph(train, test, tes_best_alpha, tes_best_beta, tes_best_gamma)


# In[91]:


################################################
# Examining the Statistical Outputs of the Model
################################################


# In[92]:


# Best Model 

last_sarimax_model = SARIMAX(train,
                             order = best_mae_order_sarimax, 
                             seasonal_order = best_mae_seasonal_order_sarimax).fit(disp = 0)


# In[93]:


last_sarimax_model.plot_diagnostics(figsize = (15,7))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




