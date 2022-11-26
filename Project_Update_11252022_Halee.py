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

# Changes 11/24/22
# 1. Remove outliers from price. 
# 
# 2. Updated "yr_renovated" to Yes/No. Only ~900 with yes. 
# 
# 3. Converted year built to age. Added column.
# 
# 4. Normalized outcome variable with square root transformation instead of log. Original data was positively skewed, and then log transformed was negatively skewed so log was not the right method.  
# 
# 5. Reduced dimensions for lat/long location groups. Added a 3X3 grid to compare. There were 3 long/lat groups that only ontained 1 house. By reducing dimensions, it grouped everything so each area is better represented. When looking at the bar chart, there are 4 distinct price levels. I think we can reduce the location dimensions even further into 3 groups (North, Central, South). The prices differences go North-South more so than East-West. This will help with final model to reduce number of dummy variables.
# 
# 6. Updated numerical/categorical/ordinal lists of variables. 
# 
# 7. **Most of our variables are discrete. We cannot compute mean and standard deviation for them. Udated missing values to fill with median. Use PMF for distributions. Median house prices for Washington state listed here: https://ofm.wa.gov/washington-data-research/statewide-data/washington-trends/economic-trends/median-home-price. 2014 = 267,600 and 2015 = 289,100. Maybe we can use this as some sort of cut-off. Add binary variable of above median or below median?? Combine with location?? ** 
# 
# 8. Ordinal variables can be used in correlations. Added them to the numerical correlation matrix. However, ordinal data is not 100% valid to use in linear regression. Grade is correlated with price, but not rating. 
# 
# 9. We cannot use price/sqft for a predictor variable. This would mean that we are using a dimension of our dependent variable inside our model, which is invalid. Just stick to continuous measurement of sqft living separate from price. Commented out this section and updated price vs. price/log histogram display.  
# 
# 10. Tested Spearman rank correlation between ordinal variables (grade and rating). No relationship found. 
# 
# 11. Added Chi-Square tests for categorical variables to test relationship with price, location, and other categories. Used log-normal price data for test. Turned "price" into categorical with 6 bins to test relationships. 6 bins = 3 SD on each side, so basically 100% of data. The year 2014 contains months from only May-Dec, and 2015 contains months from January to May. This means we cannot use "year" for any meaningful comparison. Together they create a full year of data. Confirmed with x2 test. Also tested month on its own. Month shows a difference in x2 test. Formal test of Vicky's percentage differences.  
# 
# 12. Normalized sqft_living before modeling with square root transformation.
# 
# 13. Testing relationship between all predictor variables with x2 conversion. 
# 
# 14. Final proposed predictor variables: sqft_living - log, location, yr_sold.
# 
# 15. Ran multinomial linear regression. price_log ~ sqft_log + locations(dummy) + yr_sold + e
# 
# 16. Added data visualizations.
# 
# 17. Added conclusion. 

# ## 1. Introduction

# ## 2. Data Importing and Cleaning

# * Import dataset and describe characteristics such as dimensions, data types, file types, and import methods used
# * Clean, wrangle, and handle missing data
# * Transform data appropriately using techniques such as aggregation, normalization, and feature construction
# * Reduce redundant data and perform need-based discretization

# In[155]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
from scipy.stats import spearmanr


# ### Import data, data characteristics

# In[156]:


dataset = pd.read_csv(r'C:\Users\halee\Downloads\house_sales.csv')
dataset.shape


# In[5]:


dataset_original = pd.read_csv(r'C:\Users\halee\Downloads\house_sales.csv')


# In[157]:


dataset.info()


# In[8]:


dataset.head()


# ### Clean and Handle Missing Data

# #### Clean up date and convert to date_time unit

# In[159]:


from datetime import datetime
dataset['date'] = pd.to_datetime(dataset['date'], format='%Y/%m/%d')

#Generate new columns containing the month and the year column

dataset.shape


# #### Remove values that don't reflect normal houses, that have zero bedrooms, or not full bathrooms

# In[160]:


#am also considering filling them out with average, but for now, this. 
dataset = dataset[dataset['bedrooms']!=0]
dataset = dataset[dataset['bathrooms'] > 0.5]

dataset.shape


# #### Exploring Potential Outliers

# In[161]:


#ignore mean and std
dataset.describe()


# In[162]:


dataset['price'].mode()


# In[163]:


dataset['sqft_lot'].mode()


# In[164]:


dataset['sqft_living'].mode()


# ### Check and Handle Missing Data

# In[165]:


dataset.isnull().sum()


# In[13]:


# A quick way to clean data, will be replaced later
#dataset = houses.dropna()

#Removing houses that have zero bedrooms or bathrooms; does not reflect typical home.


# In[166]:


dataset.corr()


# #### Fill in missing values in the sqft_living column 

# In[170]:


sqft_above = dataset['sqft_above']
sqft_basement = dataset['sqft_basement']
sqft_living = sqft_above + sqft_basement
dataset.loc[dataset['sqft_living'].isnull(),'sqft_living']=sqft_living

sns.scatterplot(data=dataset, x="sqft_living", y="sqft_living15")


# #### Fill in missing values in the bedrooms column

# In[171]:


#check which value correlates most (sqft_living)
#dataset.corr()
#sns.scatterplot(data=dataset, x="bedrooms", y="sqft_living")


#after viewing the scatterplot, can be seen that the relationship is not quite linear.
#updated to median
bedrooms_avg = dataset['bedrooms'].median()
dataset.loc[dataset['bedrooms'].isnull(),'bedrooms']=bedrooms_avg


# #### Fill in missing values in the sqft_lot category

# In[174]:


#even though high correlation coefficient, no clear relationship. therefore, also using averages.

sqft_lot_avg = dataset['sqft_lot'].median()
dataset.loc[dataset['sqft_lot'].isnull(),'sqft_lot']=sqft_lot_avg

sns.scatterplot(data=dataset, x="sqft_lot15", y="sqft_lot")
#new pattern popped out showing something for 15,000


# #### Fill in missing values in the bathrooms column

# In[175]:


#updated to median
bathrooms_avg = dataset['bathrooms'].median()
dataset.loc[dataset['bathrooms'].isnull(),'bathrooms']=bathrooms_avg

sns.scatterplot(data=dataset, x="bathrooms", y="sqft_living")


# #### Removing outliers 
# * After doing df.describe(), we can see that the biggest outliers are in the bedroom, sqft_lot, and sqft_living categories

# In[176]:


#highest bedroom value was 33. After further investigation, looks like data entry mistake. fill with avg
dataset['bedrooms'] = dataset['bedrooms'].replace([33],bedrooms_avg)


# In[180]:


#removed outliers with large sqft_lots
q1 = dataset['sqft_lot'].quantile(.25)
q2 = dataset['sqft_lot'].quantile(.5)
q3 = dataset['sqft_lot'].quantile(.75)
q4 = dataset['sqft_lot'].quantile(.1)
iqr = q3 - q1

#instead of doing standard 1.5iqr, did 1.65 so we did not lose too much data
#upperlimit = q3 + (1.65*iqr)
#lowerlimit = q1 - (1.65*iqr)

#tested with 1.5
upperlimit = q3 + (1.5*iqr)
lowerlimit = q1 - (1.5*iqr)

dataset = dataset[dataset['sqft_lot']<upperlimit]
#lowerlimit is less than zero so don't need to filter because no occurence of negative values

dataset['sqft_sqrt'] = np.sqrt(dataset['sqft_living'])

plt.figure(1)
plt.title('Boxplot')
sns.boxplot(data = dataset[['sqft_sqrt']])

plt.figure(2)
plt.title('Distribution of Sqft_Sqrt')
sns.distplot(dataset['sqft_sqrt'])

plt.figure(3)
plt.title('Distribution of Sqft_Living')
sns.distplot(dataset['sqft_living'])


# In[154]:


#removed outliers in sqft_living after log transformation
q1 = dataset['sqft_living'].quantile(.25)
q2 = dataset['sqft_living'].quantile(.5)
q3 = dataset['sqft_living'].quantile(.75)
q4 = dataset['sqft_living'].quantile(.1)

#iqr = q3 - q1

#instead of doing standard 1.5iqr, did 1.65 so we did not lose too much data
#upperlimit = q3 + (1.6*iqr)
#lowerlimit = q1 - (1.6*iqr)

#tested with 1.5
#upperlimit = q3 + (1.5*iqr)
#upperlimit
#lowerlimit = q1 - (1.5*iqr)
#lowerlimit

#dataset['price_sqrt'] = np.sqrt(dataset['price'])


#dataset = dataset[dataset['sqft_living']<upperlimit]
#sns.scatterplot(data=dataset, x="sqft_living", y="sqft_log")


# In[181]:


#removed outliers in price
q1 = dataset['price'].quantile(.25)
q2 = dataset['price'].quantile(.5)
q3 = dataset['price'].quantile(.75)
q4 = dataset['price'].quantile(.1)

iqr = q3 - q1

#instead of doing standard 1.5iqr, did 1.65 so we did not lose too much data
#upperlimit = q3 + (1.6*iqr)
#lowerlimit = q1 - (1.6*iqr)

#tested with 1.5
upperlimit = q3 + (1.5*iqr)
lowerlimit = q1 - (1.5*iqr)


dataset = dataset[dataset['price']<upperlimit]

dataset['price_sqrt'] = np.sqrt(dataset['price'])

plt.figure(1)
plt.title('Boxplot')
sns.boxplot(data = dataset[['price_sqrt']])

plt.figure(2)
plt.title('Distribution of Price_Sqrt')
sns.distplot(dataset['price_sqrt'])

plt.figure(3)
plt.title('Distribution of Price')
sns.distplot(dataset['price'])


# ### Drop redundant data

# In[182]:


# To ignore the warning message
warnings.filterwarnings('ignore')

#dropping the earliest entry of each duplicate ID 
dataset.sort_values(by=['id', 'date'], inplace=True)
#keeping the entry with the most recent date
dataset = dataset.drop_duplicates(subset=['id'], keep='last')
#houses.drop(['id'], axis=1, inplace = True)


# ### Feature Construction and Discretization

# #### Add columns for year sold and month sold

# In[183]:


dataset['yr_sold'] = pd.DatetimeIndex(dataset['date']).year
dataset['month_sold'] = pd.DatetimeIndex(dataset['date']).month


# **Add column to change year renovated to binary variable**

# In[261]:


#update dataset to Yes/No for renovated
dataset["renovate"] = pd.cut(dataset["yr_renovated"],2,labels = ['No','Yes'])


# In[83]:


#update year built to age
#import datetime
#t1 = dataset['yr_sold']
#t2 = dataset['yr_built']
#house_age = t1 - t2
#dataset['house_age'] = house_age


# #### Generate Categories for lat, long, and latitude + longitude

# In[185]:


dataset['lat_cat'] = pd.cut(dataset['lat'],3,labels = ['south', 'central', 'north'])
dataset.head()

dataset['long_cat'] = pd.cut(dataset['long'],5,labels = ['west', 'midwest', 'central', 'mideast', 'east'])
dataset.head()

lat_cat = dataset['lat_cat']
long_cat = dataset['long_cat']

#aggregate the latitude and longitude together for a final location category
dataset['loc_cat'] = dataset[['lat_cat', 'long_cat']].agg('-'.join, axis=1)

location_category = dataset['loc_cat']
price = dataset['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# In[186]:


dataset['lat_cat'] = pd.cut(dataset['lat'],3,labels = ['south', 'central', 'north'])
dataset.head()

dataset['long_cat'] = pd.cut(dataset['long'],3,labels = ['west', 'central', 'east'])
dataset.head()

lat_cat = dataset['lat_cat']
long_cat = dataset['long_cat']

#aggregate the latitude and longitude together for a final location category
dataset['loc_cat'] = dataset[['lat_cat', 'long_cat']].agg('-'.join, axis=1)


location_category = dataset['loc_cat']
price = dataset['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# ### Test only lattitude. 

# In[187]:


location_category = dataset['lat_cat']
price = dataset['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# ### Test only longitude.

# In[188]:


location_category = dataset['long_cat']
price = dataset['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# ### Longitude as 2 categories. 

# In[189]:


dataset['long_cat_2'] = pd.cut(dataset['long'],2,labels = ['west', 'east'])
dataset.head()

location_category = dataset['long_cat_2']
price = dataset['price']
#plot just to see
location_barplot = sns.barplot(x=location_category, y=price)
#makes it so the labels don't run into each other
location_barplot.set_xticklabels(location_barplot.get_xticklabels(), rotation = 45, horizontalalignment = 'center')


# ## 3. Exploratory Data Analysis and Visualization

# * Identify categorical, ordinal, and numerical variables within the data
# * Provide measures of centrality and distribution with visualizations
# * Diagnose for correlations between variables and determine independent and dependent variables
# * Perform exploratory analysis in combination with visualization techniques to discover patterns and features of interest

# In[190]:


# Identify categorical, ordinal, and numerical variables within the original data
numerical_variables = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'condition', 'grade', 
                      'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']
ordinal_variables = ['condition', 'grade']
categorical_variables = ['date', 'waterfront', 'view', 'zipcode', 'lat', 'long']


discrete_vars = ['price', 'bedrooms', 'bathrooms', 'floors', 'condition', 'grade',
                 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_living15']


# In[191]:


#updated lists of variables

num_vars_update = ['price_sqrt', 'bedrooms', 'bathrooms', 'sqft_sqrt', 
                       'floors', 'condition', 'grade', 'yr_built']

cat_var_update = ['lat_cat', 'long_cat', 'loc_cat', 'yr_sold', 'month_sold', 'renovate']


# ### Statistical and Correlation Analysis

# From the statistical analysis table and the correlation heatmap, we see:
# * some redundant variables, such as 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', they can be removed.
# * The house with 33 bedrooms should be treated as outlier. Therefore, it can be removed.

# In[192]:


numerical_subset_1 = dataset[numerical_variables]


# In[193]:


# Find the statistics for the numerical variables
round(numerical_subset_1.describe(), 2)


# In[194]:


# Find correlation coefficients of among the numerical variables
round(numerical_subset_1.corr(), 2)


# In[195]:


# Heatmap for correlations from original data
plt.figure(figsize=(12, 8))
plt.title("Heat Map of Correlation Coefficients", fontsize=16)
sns.heatmap(numerical_subset_1.corr(), cmap = 'coolwarm', fmt = '.1f', linewidths = 1, annot = True)


# ## **Update variables and redo matrix**

# ### Adding New Columns

# In[196]:


# Add two new columns 'price_per_sqft' and 'price_log' to facilitate analysis
#houses['price_per_sqft'] = houses['price'] / houses['sqft_living']
#numerical_subset['price_per_sqft'] = houses['price'] / houses['sqft_living']
#numerical_subset['price_log'] = np.log(houses['price'])
#houses.head()


# In[197]:


# Subset numerical fields
numerical_subset_2 = dataset[num_vars_update]


# In[198]:


# Find the statistics for the numerical variables
round(numerical_subset_2.describe(), 2)


# In[199]:


# Find correlation coefficients of among the numerical variables
round(numerical_subset_2.corr(), 2)


# In[200]:


# Heatmap for correlations
plt.figure(figsize=(12, 8))
plt.title("Heat Map of Correlation Coefficients", fontsize=16)
sns.heatmap(numerical_subset_2.corr(), cmap = 'coolwarm', fmt = '.1f', linewidths = 1, annot = True)


# ### Data Wrangling (Will be merged or replaced)

# In[201]:


# Remove some redundant variables
#warnings.filterwarnings('ignore')
#dataset.drop(['sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15'], axis=1, inplace = True)
#numerical_subset.drop(['sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15'], axis=1, inplace = True)

# Drop the outlier with the extreme number of bedrooms.tagged this out because we 
#already changed it to 3
#dataset.drop(dataset[dataset.bedrooms > 20].index, inplace = True)
#dataset.head()


# ### Redo Correlation Analysis

# ### Histogram Distribution of the Dependent Variable

# In[99]:


# Separate Prices into a new variable
prices = dataset['price']

# Define column lists for plotting
bar_cols = ['view', 'waterfront', 'condition', 'grade', 'bedrooms', 'bathrooms']
scatter_cols = ['yr_built', 'yr_renovated', 'sqft_living', 'sqft_lot', 'bathrooms', 'bedrooms', 'grade', 'floors']
pair_cols = ['sqft_living', 'sqft_lot', 'grade', 'condition']


# In[100]:


# Display Histogram for Prices
plt.figure(1)
plt.title('Distribution of Prices')
sns.distplot(prices, kde=False, fit=stats.norm)

plt.figure(2)
plt.title('Distribution of Price Log')
sns.distplot(dataset['price_log'], kde=False, fit=stats.norm)


# In[129]:


plt.figure(1)
plt.title('Distribution of Sqft_Living15')
sns.distplot(dataset['sqft_living15'])

plt.figure(2)
plt.title('Distribution of Sqft_Living')
sns.distplot(dataset['sqft_living'])

plt.figure(3)
plt.title('Distribution of Sqft_Lot15')
sns.distplot(dataset['sqft_lot15'], kde=False, fit=stats.norm)

plt.figure(4)
plt.title('Distribution of Sqft_Lot')
sns.distplot(dataset['sqft_lot'], kde=False, fit=stats.norm)


# ### Distribution Analysis with Box Plot

# In[101]:


# Create box plots

plt.figure(1)
plt.title('Boxplot')
sns.boxplot(data = dataset[['price_log']])

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

# In[202]:


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


# In[203]:


plt.figure(figsize = (18, 6))
values = dataset['zipcode'].value_counts()
pd.Series.sort_index(values, inplace=True)
indexes = list(values.index)
plt.bar(indexes, values)
plt.title("House Distribution by Zipcode", fontsize = 16)


# ### Scatter Plots for Numerical Variables vs. Price

# In[204]:


#Create scatter plots
fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(18,8))
fig.suptitle("Features vs. Prices",fontsize=16)

for n in range(len(scatter_cols)):
    i = 0 if n < 4 else 1  
    j = n % 4;
    axes[i][j].scatter(dataset[scatter_cols[n]], dataset['price_sqrt'])
    axes[i][j].set_xlabel(scatter_cols[n])
    
axes[0][0].set_ylabel("Prices")
axes[1][0].set_ylabel("Prices")

plt.show()


# ### Joint Plot to show Distribution by Latiture and Longitude

# In[206]:


#flipped the lat and long to reperesent map -v 11/19/2022
sns.jointplot(dataset['long'], dataset['lat'], size= 8)


#east locations are "location" outliers 


# # Ordinal Correlation & Tests of Association (categorical)

# Spearmans Rank Correlation Coefficient

# In[208]:


#measure of collinearity for ordinal variables
data_ord = pd.DataFrame(dataset[ordinal_variables])

scc_1, p = spearmanr(data_ord['condition'],data_ord['grade'])
print('scc_1, p = %.3f' % scc_1)


#weak negative correlation between each other


# ### Convert 'price_sqrt' into categorical bins for x2 testing. Bins = 6 (3 SD on each side)

# In[213]:


#cut dataset into 6 bins for price_log
dataset['price_sqrt_cat'] = pd.cut(dataset['price_sqrt'], 6)
dataset.head()


# ### Price and lattitude location.

# In[227]:


#chisq_freq_1_a = pd.crosstab(dataset['price_sqrt_cat'], 
                           #dataset['loc_cat'], 
                           #margins=True, margins_name='Total')
#chisq_freq_1_a

#run test
#stats.chi2_contingency(chisq_freq_1_a)

#not valid


chisq_freq_1_b = pd.crosstab(dataset['price_sqrt_cat'], 
                           dataset['lat_cat'], 
                           margins=True, margins_name='Total')
chisq_freq_1_b

#run test
stats.chi2_contingency(chisq_freq_1_b)

#stat sig, but df and x2 are far apart


# ### Price and renovation  status.

# In[228]:


#create frequency table for price bins and renovation status
chisq_freq_2 = pd.crosstab(dataset['price_sqrt_cat'].astype('category'), 
                           dataset['renovate'], 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_2
stats.chi2_contingency(chisq_freq_2)

### valid relationship between renovation and price


# ### Price and year.

# In[230]:


chisq_freq_3 = pd.crosstab(dataset['price_sqrt_cat'].astype('category'), 
                           dataset['yr_sold'].astype('category'), 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_3
stats.chi2_contingency(chisq_freq_3)

#not valid, no difference


# ### Price and month.

# In[231]:


chisq_freq_4 = pd.crosstab(dataset['price_sqrt_cat'].astype('category'), 
                           dataset['month_sold'].astype('category'), 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_4
stats.chi2_contingency(chisq_freq_4)


#valid relationship between price and month


# ## Check for Multicolinearity between Predictor Variables

# ### Change sqft_sqrt to categorical bins. 

# In[233]:


#cut square foot living  into 6 bins
dataset['sqft_sqrt_cat'] = pd.cut(dataset['sqft_sqrt'], 6)

#check dataset
dataset.head()


# ### Square feet and lattitude location. 

# In[240]:


#location and sqft_sqrt

chisq_freq_5 = pd.crosstab(dataset['sqft_sqrt_cat'].astype('category'), 
                           dataset['lat_cat'].astype('category'), 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_5
stats.chi2_contingency(chisq_freq_5)

#valid relationship


# ### Square feet and month sold

# In[245]:


#sqft_sqrt and month

chisq_freq_6 = pd.crosstab(dataset['sqft_sqrt_cat'].astype('category'), 
                           dataset['month_sold'].astype('category'), 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_6
stats.chi2_contingency(chisq_freq_6)

#not valid


# ### Square feet and renovation.

# In[246]:


#sqft_sqrt and renovate

chisq_freq_7 = pd.crosstab(dataset['sqft_sqrt_cat'].astype('category'), 
                           dataset['renovate'].astype('category'), 
                           margins=True, margins_name='Total')
#test chi square
chisq_freq_7
stats.chi2_contingency(chisq_freq_7)

#not valid relationship


# ### Location category and month sold.

# In[247]:


#create frequency table
#chisq_freq_8_a = pd.crosstab(dataset['loc_cat'], 
                           #dataset['month_sold'].astype('category'), 
                           #margins=True, margins_name='Total')
#chisq_freq_8_a

#run chi square test
#stats.chi2_contingency(chisq_freq_8_a)

#not valid, expected values below 5


chisq_freq_8_b = pd.crosstab(dataset['lat_cat'], 
                           dataset['month_sold'].astype('category'), 
                           margins=True, margins_name='Total')
chisq_freq_8_b

#run chi square test
stats.chi2_contingency(chisq_freq_8_b)

#VALID relationship


# ### Location category and renovate.

# In[250]:


#create frequency table
#chisq_freq_9_a = pd.crosstab(dataset['loc_cat'], 
                           #dataset['renovate'], 
                           #margins=True, margins_name='Total')
#chisq_freq_9_a

#run chi square test
#stats.chi2_contingency(chisq_freq_9_a)

#not valid, expected values below 5


chisq_freq_9_b = pd.crosstab(dataset['lat_cat'], 
                           dataset['renovate'], 
                           margins=True, margins_name='Total')
chisq_freq_9_b

#run chi square test
stats.chi2_contingency(chisq_freq_9_b)

#VALID relationship


# ###  Month sold and renovation.

# In[340]:


chisq_freq_10 = pd.crosstab(dataset['month_sold'], 
                           dataset['renovate'], 
                           margins=True, margins_name='Total')
chisq_freq_10

#run chi square test
#stats.chi2_contingency(chisq_freq_10)

#no relationship


# ## Based on associations tests, location is associated with multiple other variables which causes multicolinearity concerns. Since month is a predictor with 12 levels, it needs to be reduced into seasons. 

# Update month sold into 4 groups.

# In[ ]:





# ## 4. Data Modeling and Analytics

# * Determine the need for a supervised or unsupervised learning method and identify dependent and independent variables
# * Train, test, and provide accuracy and evaluation metrics for model results
# 

# ### The model for data requires a supervised learning method: multinomial linear regression.
# 
# * The outcome variable is price (price_sqrt).
# * The predictor variables are : sqft_sqrt, month sold, and renovation status. 

# In[168]:


from platform import python_version
python_version()


# ## Creating Dummy Variables

# In[341]:


#dataset['renovate_code'] = pd.to_numeric(dataset['renovate'].replace(['Yes', 'No'], [1, 0], inplace=True))
#dataset.head(50)

ren_dum = pd.get_dummies(dataset['renovate'], prefix = 'ren')
ren_dum


# In[342]:


month_dum = pd.get_dummies(dataset['month_sold'], prefix = 'mon')
month_dum


# In[306]:


dataset_final = pd.concat([dataset, month_dum, ren_dum], axis = 1)
dataset_final.head(50)


# In[307]:


np.dtype(dataset_final['ren_1'])


# In[300]:


pd.to_numeric(dataset_final['mon_1'])

#pd.to_numeric(dataset_final['mon_2'])
#pd.to_numeric(dataset_final['mon_3'])
#pd.to_numeric(dataset_final['mon_4'])
#pd.to_numeric(dataset_final['mon_5'])
#pd.to_numeric(dataset_final['mon_6'])
#pd.to_numeric(dataset_final['mon_7'])
#pd.to_numeric(dataset_final['mon_8'])
#pd.to_numeric(dataset_final['mon_9'])
#pd.to_numeric(dataset_final['mon_10'])
#pd.to_numeric(dataset_final['mon_11'])
#pd.to_numeric(dataset_final['mon_12'])


# In[301]:


np.dtype(dataset_final['mon_1'])


# ## Building Model

# ### Training Data

# In[ ]:





# In[324]:


from sklearn.linear_model import LinearRegression

X = (dataset_final[['sqft_sqrt']])
#'ren_1', 'ren_2'
#'mon_1','mon_2','mon_3','mon_4','mon_5','mon_6','mon_7','mon_8','mon_9','mon_10','mon_11','mon_12']]
y = pd.array(dataset_final['price_sqrt'])

model = LinearRegression().fit(X, y)
model.score(X,y)


# In[325]:


model.coef_


# In[326]:


model.intercept_


# In[327]:


model.get_params()


# In[339]:


model.predict([[50]])


# ## 5. Conclusion

# ## References

# In[ ]:




