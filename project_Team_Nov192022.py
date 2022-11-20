#!/usr/bin/env python
# coding: utf-8

# # The House Price Prediction
# ## Team 4: Vicky Van Der Wagt, Halee Staggs, Bethany Wang

# Changes 11/19/22
#     
# *  Inserted data cleaning setps
# * Although earlier we talked about replacing missing values with item that correlates to it the most, after looking at graphs, they do not have clear linear relationships. Therefore, I used the averages for now. Will read more projects to see what else they did. 
# * Also added discretization (5 categories for longitude, 3 categories for latitude). Then added a loc_cat column (location category) which has the concatenated values. Clear relationship between loc_cat and price
# * For duplicate ids, removed the oldest entry and only kept the most recent one. 
# * Removed outliers using quartiles in iqr for sqft_living and sqft_lot since those had the most extreme values. Also concluded the 33 bedroom entry was a mistake, as the square footage was less than 2000(definitely could not fit 33 bedrooms). Therefore, replaced it same method as other bedrooms (with average bedrooms). 
# * Holding off on deleting most columns until we decide what is relevant to our models
# 
# Vicky to do (feel free to add if you find more that needs cleaning)
# * Write introduction
# * Investigate what other people did to fill in missing bedrooms, bathrooms, sqft living

# ## 1. Introduction

# ## 2. Data Importing and Cleaning

# * Import dataset and describe characteristics such as dimensions, data types, file types, and import methods used
# * Clean, wrangle, and handle missing data
# * Transform data appropriately using techniques such as aggregation, normalization, and feature construction
# * Reduce redundant data and perform need-based discretization

# In[228]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats


# ### Import data, data characteristics

# In[262]:


dataset = pd.read_csv("house_sales.csv")
dataset.shape


# In[230]:


dataset.info()


# In[231]:


print(dataset)


# In[233]:


dataset.head()


# ### Clean and Handle Missing Data

# #### Clean up date and convert to date_time unit

# In[234]:


from datetime import datetime
dataset['date'] = pd.to_datetime(dataset['date'], format='%Y/%m/%d')

#Generate new columns containing the month and the year column

dataset.shape


# #### Remove values that don't reflect normal houses, that have zero bedrooms or bathrooms

# In[235]:


#am also considering filling them out with average, but for now, this. 
dataset = dataset[dataset['bedrooms']!=0]
dataset = dataset[dataset['bathrooms']!=0]

dataset.shape


# #### Exploring Potential Outliers

# In[236]:


dataset.describe()


# ### Check and Handle Missing Data

# In[237]:


dataset.isnull().sum()


# In[238]:


# A quick way to clean data, will be replaced later
#dataset = houses.dropna()

#Removing houses that have zero bedrooms or bathrooms; does not reflect typical home.


# In[239]:


dataset.corr()


# #### Fill in missing values in the sqft_living column 

# In[240]:


sqft_above = dataset['sqft_above']
sqft_basement = dataset['sqft_basement']
sqft_living = sqft_above + sqft_basement
dataset.loc[dataset['sqft_living'].isnull(),'sqft_living']=sqft_living


# #### Fill in missing values in the bedrooms column

# In[241]:


#check which value correlates most (sqft_living)
dataset.corr()

sns.scatterplot(data=dataset, x="bedrooms", y="sqft_living")
#after viewingt the scatterplot, can be seen that the relationship is not quite linear.
#using averages instead.
bedrooms_avg = round(dataset['bedrooms'].mean(),0)
dataset.loc[dataset['bedrooms'].isnull(),'bedrooms']=bedrooms_avg


# #### Fill in missing values in the sqft_lot category

# In[242]:


sns.scatterplot(data=dataset, x="sqft_lot", y="sqft_lot15")
#even though high correlation coefficient, no clear relationship. therefore, also using averages.

sqft_lot_avg = round(dataset['sqft_lot'].mean(),0)
dataset.loc[dataset['sqft_lot'].isnull(),'sqft_lot']=sqft_lot_avg


# #### Fill in missing values in the bathrooms column

# In[268]:


#also used average
sns.scatterplot(data=dataset, x="bathrooms", y="sqft_living")
bathrooms_avg = round(dataset['bathrooms'].mean(),0)
dataset.loc[dataset['bathrooms'].isnull(),'bathrooms']=bathrooms_avg


# #### Removing outliers 
# * After doing df.describe(), we can see that the biggest outliers are in the bedroom, sqft_lot, and sqft_living categories

# In[269]:


#highest bedroom value was 33. After further investigation, looks like data entry mistake. fill with avg
dataset['bedrooms'] = dataset['bedrooms'].replace([33],bedrooms_avg)


# In[270]:


#removed outliers with large sqft_lots
q1 = dataset['sqft_lot'].quantile(.25)
q2 = dataset['sqft_lot'].quantile(.5)
q3 = dataset['sqft_lot'].quantile(.75)
q4 = dataset['sqft_lot'].quantile(.1)
iqr = q3 - q1

#instead of doing standard 1.5iqr, did 1.65 so we did not lose too much data
upperlimit = q3 + (1.65*iqr)
lowerlimit = q1 - (1.65*iqr)

dataset = dataset[dataset['sqft_lot']<upperlimit]
#lowerlimit is less than zero so don't need to filter because no occurence of negative values


# In[271]:


#removed outliers in sqft_living
q1 = dataset['sqft_living'].quantile(.25)
q2 = dataset['sqft_living'].quantile(.5)
q3 = dataset['sqft_living'].quantile(.75)
q4 = dataset['sqft_living'].quantile(.1)

iqr = q3 - q1

#instead of doing standard 1.5iqr, did 1.65 so we did not lose too much data
upperlimit = q3 + (1.6*iqr)
lowerlimit = q1 - (1.6*iqr)

dataset = dataset[dataset['sqft_living']<upperlimit]
sns.scatterplot(data=dataset, x="bedrooms", y="sqft_living")


# ### Drop redundant data

# In[255]:


# To ignore the warning message
warnings.filterwarnings('ignore')

#dropping the earliest entry of each duplicate ID 
dataset.sort_values(by=['id', 'date'], inplace=True)
#keeping the entry with the most recent date
dataset = dataset.drop_duplicates(subset=['id'], keep='last')
#houses.drop(['id'], axis=1, inplace = True)


# ### Feature Construction and Discretization

# #### Add columns for year sold and month sold

# In[256]:


dataset['yr_sold'] = pd.DatetimeIndex(dataset['date']).year
dataset['month_sold'] = pd.DatetimeIndex(dataset['date']).month


# #### Generate Categories for lat, long, and latitude + longitude

# In[257]:


dataset['lat_cat'] = pd.cut(dataset['lat'],3,labels = ['south', 'central', 'north'])
dataset.head()

dataset['long_cat'] = pd.cut(dataset['long'],5,labels = ['west', 'midwest', 'central', 'mideast', 'east'])
dataset.head()

lat_cat = dataset['lat_cat']
long_cat = dataset['long_cat']

#aggregate the latitude and longitude together for a final location category
dataset['loc_cat'] = dataset[['lat_cat', 'long_cat']].agg('-'.join, axis=1)

location_category = dataset['loc_cat']
price = houses['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# ## 3. Exploratory Data Analysis and Visualization

# * Identify categorical, ordinal, and numerical variables within the data
# * Provide measures of centrality and distribution with visualizations
# * Diagnose for correlations between variables and determine independent and dependent variables
# * Perform exploratory analysis in combination with visualization techniques to discover patterns and features of interest

# In[258]:


# Identify categorical, ordinal, and numerical variables within the data

numerical_variables = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 
                       'lat', 'long', 'sqft_living15', 'sqft_lot15']
ordinal_variables = ['condition', 'grade', 'yr_built', 'yr_renovated', 'yr_sold', 'month_sold']
categorical_variables = ['date', 'waterfront', 'view', 'zipcode', 'lat_cat', 'long_cat', 'loc_cat']
print("hi")


# ### Statistical and Correlation Analysis

# In[259]:


# Subset numerical fields
numerical_subset = dataset[numerical_variables]


# In[260]:


# Find the statistics for the numerical variables
round(numerical_subset.describe(), 2)


# In[261]:


# Find correlation coefficients of among the numerical variables
round(numerical_subset.corr(), 2)


# In[ ]:


# Heatmap for correlations
plt.figure(figsize=(12, 8))
plt.title("Heat Map of Correlation Coefficients", fontsize=16)
sns.heatmap(numerical_subset.corr(), cmap = 'coolwarm', fmt = '.1f', linewidths = 1, annot = True)


# From the statistical analysis table and the correlation heatmap, we see:
# * some redundant variables, such as 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', they can be removed.
# * The house with 33 bedrooms should be treated as outlier. Therefore, it can be removed.

# ### Data Wrangling (Will be merged or replaced)

# In[324]:


# Remove some redundant variables
warnings.filterwarnings('ignore')
#houses.drop(['sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15'], axis=1, inplace = True)
#numerical_subset.drop(['sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15'], axis=1, inplace = True)

# Drop the outlier with the extreme number of bedrooms.tagged this out because we 
#already changed it to 3
#houses.drop(houses[dataset.bedrooms > 20].index, inplace = True)
dataset.head()


# ### Adding New Columns

# In[325]:


# Add two new columns 'price_per_sqft' and 'price_log' to facilitate analysis
houses['price_per_sqft'] = houses['price'] / houses['sqft_living']
houses['price_log'] = np.log(houses['price'])

numerical_subset['price_per_sqft'] = houses['price'] / houses['sqft_living']
numerical_subset['price_log'] = np.log(houses['price'])
houses.head()


# ### Redo Correlation Analysis

# In[326]:


# Heatmap for correlations
plt.figure(figsize=(12, 8))
plt.title("Heat Map of Correlation Coefficients", fontsize=16)
sns.heatmap(numerical_subset.corr(), cmap = 'coolwarm', fmt = '.1f', linewidths = 1, annot = True)


# ### Histogram Distribution of the Dependent Variable

# In[327]:


# Separate Prices into a new variable
prices = dataset['price']

# Define column lists for plotting
bar_cols = ['view', 'waterfront', 'condition', 'grade', 'bedrooms', 'bathrooms']
scatter_cols = ['yr_built', 'yr_renovated', 'sqft_living', 'sqft_lot', 'bathrooms', 'bedrooms', 'grade', 'floors']
pair_cols = ['sqft_living', 'sqft_lot', 'grade', 'condition']


# In[328]:


# Display Histogram for Prices
plt.figure(1)
plt.title('Distribution of Prices')
sns.distplot(prices, kde=False, fit=stats.norm)

# Display Histogram for Price-Per-Sqft
plt.figure(2)
plt.title('Distribution of Price-Per_Sqft')
sns.distplot(numerical_subset['price_per_sqft'], kde=False, fit=stats.norm)

# Display Histogram for Price Log
plt.figure(3)
plt.title('Distribution of Price Log')
sns.distplot(houses['price_log'], kde=False, fit=stats.norm)


# ### Distribution Analysis with Box Plot

# In[329]:


# Create box plots

plt.figure(1)
plt.title('Boxplot')
sns.boxplot(data = dataset[['price']])

plt.figure(2)
plt.title('Boxplots')
sns.boxplot(data = dataset[['bedrooms', 'bathrooms', 'floors', 'condition', 'grade']])

plt.figure(3)
plt.title('Boxplots')
sns.boxplot(data = dataset[['sqft_living']])

plt.figure(4)
plt.title('Boxplots')
sns.boxplot(data = dataset[['sqft_lot']])


# ### Bar Plots for Categorical and Discrete Variables

# In[333]:


# Create bar plots
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(18,12))
fig.suptitle("Bar Graphs",fontsize=16)

for n in range(len(bar_cols)):
    i = 0 if n < 3 else 1  
    j = n % 3;
    values = dataset[bar_cols[n]].value_counts()
    pd.Series.sort_index(values, inplace=True)
    indexes = list(values.index)
    axes[i][j].bar(indexes, values)
    axes[i][j].set_xlabel(bar_cols[n])
    
axes[0][0].set_ylabel("Number of Houses")
axes[1][0].set_ylabel("Number of Houses")

plt.show()


# In[334]:


plt.figure(figsize = (18, 6))
values = dataset['zipcode'].value_counts()
pd.Series.sort_index(values, inplace=True)
indexes = list(values.index)
plt.bar(indexes, values)
plt.title("House Distribution by Zipcode", fontsize = 16)


# ### Scatter Plots for Numerical Variables vs. Price

# In[336]:


#Create scatter plots
fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(18,8))
fig.suptitle("Features vs. Prices",fontsize=16)

for n in range(len(scatter_cols)):
    i = 0 if n < 4 else 1  
    j = n % 4;
    axes[i][j].scatter(dataset[scatter_cols[n]], prices)
    axes[i][j].set_xlabel(scatter_cols[n])
    
axes[0][0].set_ylabel("Prices")
axes[1][0].set_ylabel("Prices")

plt.show()


# ### Joint Plot to show Distribution by Latiture and Longitude

# In[68]:


#flipped the lat and long to reperesent map -v 11/19/2022
sns.jointplot(dataset['long'], houses['lat'], size= 8)


# In[ ]:





# ## 4. Data Modeling and Analytics

# * Determine the need for a supervised or unsupervised learning method and identify dependent and independent variables
# * Train, test, and provide accuracy and evaluation metrics for model results
# 

# ## 5. Conclusion

# ## References

# In[ ]:




