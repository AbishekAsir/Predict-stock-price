#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nsepy import get_history 
from datetime import date
import pandas as pd
import os


# In[2]:


nifty_opt = get_history(symbol="NIFTY", start=date(2001,1,1), end=date(2021,11,20),
index=True)


# In[3]:


nifty_opt


# In[4]:


nifty_opt.plot.line(y="Close", use_index=True)


# In[5]:


del nifty_opt["Turnover"]


# In[6]:


nifty_opt


# In[7]:


nifty_opt["Tomorrow"] = nifty_opt["Close"].shift(-1)


# In[8]:


nifty_opt


# In[9]:


nifty_opt["Target"] = (nifty_opt["Tomorrow"] > nifty_opt["Close"]).astype(int)


# In[10]:


nifty_opt


# In[11]:


from sklearn.ensemble import RandomForestClassifier


# In[13]:


model = RandomForestClassifier(n_estimators=200,min_samples_split = 40,random_state=1)


# In[14]:


train = nifty_opt.iloc[:-100]
test = nifty_opt.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[34]:


type.test


# In[15]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)


# In[16]:


combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()


# In[17]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[18]:


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


# In[19]:


predictions = backtest(nifty_opt, model, predictors)


# In[20]:


predictions["Predictions"].value_counts()


# In[21]:



precision_score(predictions["Target"], predictions["Predictions"])


# In[22]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[23]:


horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = nifty_opt.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    nifty_opt[ratio_column] = nifty_opt["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    nifty_opt[trend_column] = nifty_opt.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]


# In[24]:



nifty_opt = nifty_opt.dropna(subset=nifty_opt.columns[nifty_opt.columns != "Tomorrow"])


# In[25]:


nifty_opt


# In[26]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# In[27]:


predictions = backtest(nifty_opt, model, new_predictors)


# In[28]:


predictions["Predictions"].value_counts()


# In[29]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[30]:



predictions["Target"].value_counts() / predictions.shape[0]


# In[31]:


predictions


# In[ ]:




