#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install pingouin 


# In[4]:


pip install termcolor


# In[5]:


import pandas as pd
import scipy.stats as stats
import numpy as np
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
from termcolor import colored
import operator

from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot


# In[6]:


# Read the data from the file
# Data from https://www.kaggle.com/datasets/piumiu/used-cars-database-50000-data-points
df = pd.read_csv('/Users/terezasaskova/Downloads/3/practiceCode3/Data_P3.csv', encoding_errors='ignore')


# In[7]:


# Check the first few lines to see the column names and type of content
df.head()


# In[8]:


# Size of the dataframe
df.shape


# In[9]:


# Remove rows with empty values

df = df.replace(r'^\s*$', float('NaN'), regex = True)
df.dropna(inplace = True)

df.shape


# In[10]:


#CHECK - TYPES of the different columns

df.dtypes


# In[11]:


# Let us adapt the dataframe (remove IDs, and make sure that columns have the right type)
del df['dateCrawled'] 
del df['name']
del df['model']
del df['dateCreated'] 
del df['nrOfPictures'] 
del df['postalCode'] 
del df['lastSeen'] 
del df['seller']
del df['offerType']

df['abtest'] = df['abtest'].astype('category')
df['vehicleType'] = df['vehicleType'].astype('category')
df['gearbox'] = df['gearbox'].astype('category')
df['fuelType'] = df['fuelType'].astype('category')
df['brand'] = df['brand'].astype('category')
df['notRepairedDamage'] = df['notRepairedDamage'].astype('category')
df['yearOfRegistration'] = df['yearOfRegistration'].astype(float)
df['powerPS'] = df['powerPS'].astype(float)
df['monthOfRegistration'] = df['monthOfRegistration'].astype('category')


# In[12]:


for index, row in df.iterrows(): 
    df.loc[index,'price']=float(row['price'].replace('$','').replace(',','')) 
    df.loc[index,'odometer']=float(row['odometer'].replace('km','').

replace(',',''))
    
df['price'] = df['price'].astype(float) 
df['odometer'] = df['odometer'].astype(float)


# In[13]:


#checking the types of the final dataframe
df.info()


# In[14]:


#DATA EXPLORATION
# Lets see the summary of the numerical columns
df.describe()


# In[15]:


#Histogram of prices

df.hist(column="price")


# In[ ]:


# Remove outliers
df=df[(np.abs(stats.zscore(df[['price','yearOfRegistration','powerPS','odometer']])) < 3).all(axis=1)]
df=df[(np.abs(stats.zscore(df[['price','yearOfRegistration','powerPS','odometer']])) < 3).all(axis=1)]

df.shape


# In[ ]:


df.describe()


# In[ ]:


#Histogram of prices

df.hist(column="price");


# In[35]:


#filter those YEARS and BRANDS of interest (important !!)

#  Filter data to set the final problem
first_year = 1987
last_year = 2018

brands = ['alfa_romeo', 'audi', 'bmw', 'chevrolet', 'chrysler', 'citroen',
'dacia','daewoo','daihatsu', 'fiat', 'ford', 'honda', 'hyundai', 'jaguar',
'jeep', 'kia','lada','lancia', 'land_rover', 'mazda', 'mercedes_benz', 'mini',
'mitsubishi','nissan', 'opel', 'peugeot', 'porsche', 'renault', 'rover', 'saab',
'seat','skoda', 'smart','subaru', 'suzuki', 'toyota', 'trabant',
'volkswagen', 'volvo']

rows_sel = df.brand.isin(brands) & ((df.yearOfRegistration >= first_year) & (df.yearOfRegistration <= last_year))

df = df.loc[rows_sel]



# In[ ]:


print('Final size of the dataframe df: ', df.shape)


# In[34]:


fig = plt.figure(1, figsize=(15, 5))
plt.subplot(1,2,1)
df['brand'].hist(orientation = 'horizontal')
plt.xlabel('brand')
plt.subplot(1,2,2)
df['yearOfRegistration'].hist(bins=100)
plt.xlabel('yearOfRegistration')

plt.show()
print('Final size of the dataframe df: ', df.shape)


# In[22]:


print('Final size of the dataframe df: ', df.shape)

# Select only numeric columns
numeric_df = df.select_dtypes(include=[float, int])

# Compute the correlation matrix and convert to float
corrMatrix = numeric_df.corr().astype(float)

# Display the correlation matrix
print(corrMatrix)


# In[23]:


sns.heatmap(corrMatrix, annot=True) 

plt.show()


# In[24]:


def explore(df, yname, xname):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.boxplot(x=xname, y=yname, data=df, ax=ax)



# In[36]:


def explore(df, yname, xname):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.boxplot(x=xname, y=yname, data=df, ax=ax)
    # sns.swarmplot(x=xname, y=yname, data=df, color='black', alpha=0.5, ax=ax)

    print('Checking normality with Shapiro-Wilk')
    print(pg.normality(data=df, dv=yname, group=xname))
    print(' ')
    
    print('Checking homoscedasticity')
    print(pg.homoscedasticity(data=df, dv=yname, group=xname, method='levene'))
    print(' ')
    
    print('Checking dependence of y on x (1-way ANOVA)')
    aov = pg.anova(data=df, dv=yname, between=xname, detailed=True)
    print(aov)
    
    pvalue = aov['p-unc'][0]
    if pvalue < 0.05:
        color = 'green'
    else:
        color = 'red'
    
    print(f'ANOVA p-value is {pvalue}, dependence detected: {color == "green"}')

  


# In[26]:


F={}
F['abtest']=explore(df,"price", "abtest")


# In[27]:


F['vehicleType']=explore(df,"price","vehicleType")


# In[28]:


F['gearbox']=explore(df,"price", "gearbox")


# In[29]:


F['monthOfRegistration']=explore(df,"price", "monthOfRegistration")


# In[30]:


F['fuelType']=explore(df,"price", "fuelType")


# In[31]:


F['brand']=explore(df,"price", "brand")


# In[32]:


F['notRepairedDamage']=explore(df,"price","notRepairedDamage")


# In[37]:


# Filter out None values
filtered_F = {k: v for k, v in F.items() if v is not None}

# Sort by value and print within the loop
for k, Fk in sorted(filtered_F.items(), key=operator.itemgetter(1), reverse=True):
    print('%s: F=%f' % (k, Fk))

# Print last values outside the loop if they were set
if filtered_F:
    last_k, last_Fk = next(reversed(sorted(filtered_F.items(), key=operator.itemgetter(1), reverse=True)))
    print('Last sorted item: %s: F=%f' % (last_k, last_Fk))


# In[38]:


#2-WAY ANOVA - relation between any of the discrete variables and the price. ANOVA is a technique of the form y=f(x) where y is continous and x is discrete.

fig, ax = plt.subplots(figsize=(8, 6))
fig = interaction_plot(
    x        = df.gearbox.astype('object'),
    trace    = df.notRepairedDamage.astype('object'),
    response = df.price,
    ax       = ax,
)


# In[39]:


fig, ax = plt.subplots(figsize=(8, 6))

fig = interaction_plot(
    x= df.gearbox.astype('object'),
    trace = df.vehicleType.astype('object'),
    response = df.price,
    ax = ax) 

pg.anova(
    data = df,
    dv = 'price',
    between = ['gearbox', 'vehicleType'], 
    detailed = True
).round(4)


# In[40]:


fig, ax = plt.subplots(figsize=(8, 6))
fig = interaction_plot(
    x        = df.gearbox.astype('object'),
    trace    = df.fuelType.astype('object'),
    response = df.price,
    ax       = ax) 

pg.anova(
data = df,
dv = 'price',
between = ['gearbox', 'fuelType'], detailed = True
).round(4)


# In[41]:


formula = "price ~ gearbox + notRepairedDamage + vehicleType + fuelType" 

lm = ols(formula, df).fit()

print(lm.summary())
print(colored("The R2 of this model is %f"%lm.rsquared_adj,"green"))


# In[42]:


df['cYear']=df['yearOfRegistration']-df['yearOfRegistration'].mean()
df['cPower']=df['powerPS']-df['powerPS'].mean()
df['cOdometer']=df['odometer']-df['odometer'].mean()
formula = "price ~ cYear + cPower + cOdometer"
lm = ols(formula, df).fit()

print(lm.summary())
print(colored("The R2 of this model is %f"%lm.rsquared_adj,"green"))


# In[43]:


formula = "price ~ cYear + cPower + cOdometer + cYear*cPower + cYear*cOdometer+ cPower*cOdometer"
lm = ols(formula, df).fit()

print(lm.summary())
print(colored("The R2 of this model is %f"%lm.rsquared_adj,"green"))


# In[44]:


formula += "+ gearbox + notRepairedDamage + vehicleType + fuelType"

lm = ols(formula, df).fit()

print(lm.summary())
print(colored("The R2 of this model is %f"%lm.rsquared_adj,"green"))


# In[ ]:





# In[ ]:





# In[ ]:




