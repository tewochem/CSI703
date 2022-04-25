#!/usr/bin/env python
# coding: utf-8

# 
# * **Created by**       : Tewodros (Teddy) Weldeamlak
# * **Student Number**   : G01323583
# * **Created date**     : 2/28/22
# 
# * **Description**      : This program will be used for the entire semester. The code started from the first assignment about "Tell me your secrets" and any other upcoming assignments will be added here or modified based on the feedback or any new assignment, any modification will be noted on the modified date and by part of this annotation. 
# *                        It is importing a forest fire dataset from excel file to pandas dataframe. 
# *                        The program is used to visualize the following three questions: 
#                               * Question 1: Question 1: What is the average temprature, relativie humdity and 
#                                             wind over certain time range (years) 
#                                             in Montesinho natural park[Cortez and Morais, 2007]?
#                               * Question 2: What is the average percipitations for each day for each month 
#                                             in Montesinho natural park[Cortez and Morais, 2007]?
#                               * Question 3: How wind and temperature related or contributes to the DC (Drought code)?
#                               * Question 4: What is the distribuition wind per day for a give time range?
# 
# * **Modified by**      : Tewodros (Teddy) Weldeamlak 
# * **Modified date**    : 3/3/22 Modidfying the existing code to add assigment for **Data munging is fun!**
#                           *  Assignment task 1
#                                   * How many rows and columns does your data have?
#                                   * What are the different data types in the dataset 
#                                     (e.g., string, Boolean, integer, floating, date/time, categorical, etc.)?
#                                   * What variables would you rename to make your visualization look better?
#                                   * Describe any missing values. Using the rule of thumb in Data Visualization Made 
#                                     Simple, would you remove those rows or columns?
#                                   * What other cleaning / prep steps would you do, based on the advice in 
#                                     Data Visualization Made Simple.
#                           * Assignment task 2: 
#                                   * Using the code examples in the Data Visualization Workshop files,
#                                     perform at least one cleaning step on your data. Use a Markdown cell to describe
#                                     what you cleaned and how. Use a code cell to write the code that performs the cleaning.
# * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 3/24/22**
# 
#                **This is assignment for Take Assessment: Visualizing correlation, comparisons, and trends.**
# 
# 
# * Use the dataset and Jupyter Notebook you've been using. 
# 
# * Submitter (you can receive a max of 8 points): 
# 
# * Task 1: Develop one question related to correlation, comparison or trends for this dataset. Write the question in a Markdown cell in your notebook.
# 
# * Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.
# 
# * Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.
# 
# * Task 4: DO NOT SUBMIT YOUR NOTEBOOK FOR THIS. Hover over the graphic in your notebook and you'll see a tiny save icon. Click on that and save the graphic as an image. Upload the image, write your question and answer in the assignment submission box, and submit everything.
# 
# * Reviewer (you can receive a max of 3 points, one per review): Open the graphic. Assign points using the criteria provided. Also provide feedback if you wish. Submit.
# 
# 
# * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 3/29/22**
# 
# * Visualizing distributions and part-to-whole
# * Loading covid data for distribution plot 
# 
# 
# * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 4/6/22**
# 
# * Visualizing geospacial aspects 
#  
#  * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 4/6/22**
# 
# * Visualizing concept and qualtative aspects 
#  
#  * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 4/6/22**
# * Take Assessment: What does that even mean? Take #2
#   
#  * **Modified by:  Teddy Weldeamlak**
# * **Modified date: 4/21/22**

# **Part 1 - Importing the libraries that are needed for now and future use.**

# In[2]:


#Importing libraries needed for the assignment (Tell me your secrets.)
 
import pandas as pd 
import pandas as DataFrame
import pandas as Grouper 
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

import numpy as np


import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 

import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# In[3]:



pd.set_option('display.width', 1000)


# In[2]:


get_ipython().system('pip3 install wordcloud')


# In[3]:


get_ipython().system('pip3 install plotly')


# **Part 2 - Loading the forestfire dataset from the local drive and adding it to python dataframe**

# In[4]:


#importing the forest fires dataset by reading the excel file and converting it to pandas dataframe 

filepath = 'D:\\CSI703_Scentific and StatisticalDataVisualzation\\Assignments\\Tell me your secretes\\forestfires.xlsx'

df = pd.read_excel(open(filepath,'rb'), sheet_name='forestfires')

#Loading covid related data for visualzation for distribution plot.

df2 = pd.read_csv('D:\\CSI678_Time Series Analysis and Forcasting\\project\\owid-covid-data.csv',parse_dates=['date'],index_col=['date'],dayfirst=True)


#Loading covid related data for visualzation for distribution plot.

dfwine = pd.read_csv( 'D:\\CSI703_Scentific and StatisticalDataVisualzation\\Assignments\\concepts or qualitative aspects\\winemag-data-130k-v2.csv', index_col=0)


# **Part 3 - Preliminary exploration of the forest fires dataset**

# In[5]:


# displaying the first five rows of dataset 

df.head(5)


# In[132]:


#displayinf last five rows of dataset
df.tail()


# In[7]:


#some info. about descriptive statistics include those that summarize the central tendency, dispersion
df.describe(include='all') 


# In[8]:


#more information about the dataframe for forest fires dataset 
df.info()


# **Part 4 - Do some clean up such as: dropping some columns or dropping missing values**

# In[9]:


#checking for any missing or null values for the entire forest fires dataset 

df.isna().sum()


# In[10]:


df2 = df.drop(['X', 'Y', 'day','month'], axis=1)
df2


# **Part 5: Data visualization that the refelect the questions for this assignment**

# In[11]:


#creating correlation matrix 

matrix = df2.corr().round(3)

sns.heatmap(matrix, annot=True, vmax=1 , vmin=-1, center=0, cmap='vlag')


plt.title("Correlation matrix of forest fires with meteorological data")


plt.xlabel("Forest fires attributes")


plt.ylabel("Forest fires attributes")


plt.show()


# **Question 1: What is the average temprature, relativie humdity and wind over certain time range (years) in Montesinho natural park[Cortez and Morais, 2007]?**

# In[12]:


sns.violinplot(x=df['temp'])


# In[13]:


sns.violinplot(x=df['RH'])


# In[14]:


sns.violinplot(x=df['wind'])


# In[15]:


sns.violinplot(x=df['rain'])


# In[16]:


# This is to see the average temp over days

df3 = sns.violinplot(x="day", y="temp", data=df)


# **Question 2: What is the average percipitations for each day for each month in Montesinho natural park[Cortez and Morais, 2007]?**

# In[17]:


all_month_year_df = pd.pivot_table(df, values="rain",
                                   index=["day"],
                                   columns=["month"],
                                   fill_value=0,
                                   margins=True)
named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]] # name months
all_month_year_df = all_month_year_df.set_index(named_index)


# In[18]:


print(all_month_year_df)


# In[19]:


ax = sns.heatmap(all_month_year_df, cmap='RdYlGn_r',
                 robust=True,
                 fmt='.2f',
                 annot=True,
                 linewidths=.5,
                 annot_kws={'size':11},
                 cbar_kws={'shrink':.8,
                           'label':'Precipitation(mm)'})                       
    
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
plt.title('Average Precipitations', fontdict={'fontsize':18},    pad=14);


# **Question 3: How wind and temperature related or contributes to the DC (Drought code)?**

# In[20]:


g = sns.pairplot(df[['temp','DC','wind']], kind="kde")

g.fig.suptitle("Wind and temperature correlation to DC (Drought code)")


# **Question 4: What is the distribuition wind per day for a give time range?**

# In[21]:


fig = plt.figure(figsize =(11, 7))

sns.boxplot(x='day', y='wind', data=df)

plt.title("Distribuition wind per day for a give time range")

plt.show()


# In[22]:


# This is addtional plot to see the distribution of wind per day per month 
fig = plt.figure(figsize =(20, 7))
 
# Creating plot

sns.boxplot(x='day', y='wind',hue='month', data=df)

# show plot
plt.show()


# *  Assignment task 1
#                                   * How many rows and columns does your data have?
#                                   * What are the different data types in the dataset 
#                                     (e.g., string, Boolean, integer, floating, date/time, categorical, etc.)?
#                                   * What variables would you rename to make your visualization look better?
#                                   * Describe any missing values. Using the rule of thumb in Data Visualization Made 
#                                     Simple, would you remove those rows or columns?
#                                   * What other cleaning / prep steps would you do, based on the advice in 
#                                     Data Visualization Made Simple.

# **How many rows and columns does your data have?**
# 
# * Using the info() method of pandas.DataFrame helps to display the rows and columns, in additon to other information
#   such as can total memory usage, the data type of each column, and the number of
#   non-NaN elements.

# In[23]:


# There are 517 rows and 13 columns in forest fire dataset 
df.info()


#  **What are the different data types in the dataset (e.g., string, Boolean, integer, floating, date/time, categorical, etc.)?**

# In[24]:


# we can use info() method to know the diffrent data types in forest fire dataset, it has integer, 
#float and object or categorical (day and month columns )

df.info()


#  **What variables would you rename to make your visualization look better?**

# In[25]:


# I will rename the following colomns 
df.rename(columns={
    "month": "Month",
    "day": "Day",
    "FFMC": "Fine Fuel Moisture Code",
    "DMC": "Duff Moisture Code",
    "DC":" Drought Code",
    "ISI":"Initial Spread Index",
    "temp":"Temperature",
    "RH":"Relative Humidity",
    "wind":"Wind",
    "rain":"Rain",
    "area":"Area"
},
          inplace=True)

print('After:', df.columns)


# In[26]:


print(df)


# In[27]:


sns.violinplot(x=df['Relative Humidity'])


#  
#  
#  
#  
#  
#  **Describe any missing values. Using the rule of thumb in Data Visualization Made Simple, would you remove those rows 
#    or columns?**

# In[28]:


#checking for any missing or null values for the entire forest fires dataset, there is no any missing value in forest fire dataset 

df.isna().sum()


# * The forest fire dataset has no any missing values, I have uploaded it as xlsx file from the blackboard. 
#   There are many zero values. 
# * However, if the dataset has missing values i would the following codes below.
# * If the number of missing values is <=5%, I would remove the row. 
# * I will also try to apply imputation using models 
# * Approperate guess is another option 
# * Using mean 
# * Or, use interpolate() method to performs linear interpolation at missing data points.

# In[29]:


#filling missing values: fillna
df.fillna(0)


# In[30]:


df['Rain'].fillna("missing")


# In[31]:


# Using Fill gaps forward or backward method 
df.fillna(method="pad")


# In[32]:


df.fillna(df.mean())


# **What other cleaning / prep steps would you do, based on the advice in Data Visualization Made Simple.**

# * Cleaning: 
#        * Removing unnecessary variables
#        * Deleting duplicate rows/observations
#        * Addressing outliers or invalid data
#        * Dealing with missing values
#        * Standardizing or categorizing values
#        * Correcting typographical errors
# 
# * Prepration: 
#       * Formatting columns appropriately (numbers are treated as numbers, dates as dates)
#       * Convert values into appropriate units
#       * Filter your data to focus on the specific data that interests you.
#       * Group data and create aggregate values for groups (Counts, Min, Max, Mean, Median, Mode)
#       * Extract values from complex columns
#       * Combine variables to create new columns
# 

# **Assignment task 2:** 
# 
# * Using the code examples in the Data Visualization Workshop files,perform at least one cleaning step on your data. 
#   Use a Markdown cell to describe what you cleaned and how. Use a code cell to write the code that performs the cleaning.
#                                     

# **Cleanup 1:**
#  **Removing unnecessary variables**
#  * For instance during data visualization step on the first assignment I have used correlation matrix, and I was not interested to the following variables: X , Y, day and month on the correlation matrix. Therefore, I have dropped them from my dataset.
# 

# * Step1: dropping unnecessary variables that we don't need for correlation matrix and saving the dataset into a diffrent dataset name (dfremov)

# In[33]:


# Removing unnecessary variables
dfremov = df.drop(['X', 'Y', 'day','month'], axis=1)


# **Step 2: checking if the variables are dropped**

# In[ ]:


print(dfremov)


# **Step 3: Apply correlation matrix**

# In[ ]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(dfremov.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

# dpi - sets the resolution of the saved image in dots/inches
# bbox_inches - when set to 'tight' - does not allow the labels to be cropped
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')


# **Cleanup 2:**
# 
# **Addressing outliers or invalid data**
# 
# * Addressing outliers or invalid data needs a good understanding of the dataset, for instance, as illustrated below using the box plot using univariate method and also using Z-score treatment, the Initial Spread Index (ISI) which is a score that correlates with fire velocity spread was taken as an example to demonstrate addressing outlier issue. It is important to understand the background/research conducted on forest fire in Montesinho natural park [Cortez and Morais, 2007] before making any decision to remove record due to an outlier. It is assumed that the feature is normally or approximately normally distributed.
# 
# 
# 

# In[ ]:


plt.figure(figsize=(5,5))
sns.boxplot(y='ISI',data=df)


# In[ ]:


ISI = df.ISI.unique()


# In[ ]:


ISI.sort()


# In[ ]:


print(ISI)


# In[ ]:


#Plot the Distribution plots for the features
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['ISI'])


# In[ ]:


#Finding the Boundary Values
print("Highest allowed",df['ISI'].mean() + 3*df['ISI'].std())
print("Lowest allowed",df['ISI'].mean() - 3*df['ISI'].std())


# In[ ]:


# Finding the Outliers
df[(df['ISI'] > 22.7) | (df['ISI'] < -4.66)]


# In[ ]:


# Trimming of Outliers

new_df = df[(df['ISI'] < 22.7) & (df['ISI'] > -4.66)]
new_df


# In[ ]:


# Capping on Outliers
upper_limit = df['ISI'].mean() + 3*df['ISI'].std()
lower_limit = df['ISI'].mean() - 3*df['ISI'].std()


# In[ ]:


# Now, apply the Capping

df['ISI'] = np.where(
    df['ISI']>upper_limit,
    upper_limit,
    np.where(
        df['ISI']<lower_limit,
        lower_limit,
        df['ISI']
    )
)


# In[ ]:


# Now see the statistics using “Describe” Function
df['ISI'].describe()


# In[ ]:





# In[ ]:


print(df)


# **Task 1: Develop one question related to correlation, comparison or trends for this dataset. Write the question in a Markdown cell in your notebook**

# **In reference to the dataset that was given in the class about forest fires, let's say a government environmental agency wanted to know the correlation of major metrological condition such as temperature, wind and relative humidity with respect to the area burned by the fires in Montesinho natural park. The agency wanted to make a decision on resources management such us air tanker or ground crews in fighting the fires, therefore, the question that the agency wanted to know is: 
# Which metrological condition such as temperate, rain, relative humidity or wind correlated positively or negatively to the area burned in Montesinho natural park so that they can get some preliminary clue on the allocation resources? ***
# 

# **Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.**

# In[ ]:


sns.set(rc = {'figure.figsize':(15,8)})

# plotting correlation heatmap
dataplot = sns.heatmap(df[['Area', 'Wind','Temperature','Rain','Relative Humidity']].corr(), cmap="YlGnBu", annot=True)

dataplot.set_title('Correlation Matrix Plot for Metrological condtions and Forest area burned')

# displaying heatmap
plt.show()


# **Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.**

# It is possible to see from correlation matrix plot that those metrological conditions such as wind and temperature are positively correlated to the area burned where as rain and relative humidity are negatively correlated. Therefore, this correlation matrix plot will give some preliminary clue of the major metrological condition that cause forest fires from those mentioned factors above. The idea is what metrological condition such temperate, wind, rain and relative humidity or a combination two or more of those variables causes larger burned areas. If the burned area is small by those factors then there is a need of small resources that the government has to allocate to fight the fire. This is only a preliminary assessment but to fully answer the question other additional plot such as streamgraph or stack area might need to be incorporated to get a full insight. However, the correlation matrix at least gives the higher mangers the need to allocate more rescores during windy or high temperature weather condition as compared to rain or relative humidity condition. I also add additional variables on the matrix plot to see what other variables contribute high impact on the forest fires but I have limited variables in the forest fire dataset. 

# In[39]:


df2.head()


# **Task 1: Develop one question related to distributions or part-to-whole for this dataset. Write the question in a Markdown cell in your notebook.**

# * How was the distribution of new covid19 cases for the following countries: Italy, Sourth Africa and Mexico between 2021-01-01 and 2021-02-01?

# In[106]:


df2usa = df2[(df2.index >= '2021-01-01') & (df2.index <= '2021-02-01') & (df2['location'].isin([ 'Italy','South Africa','Mexico']))]
#'Germany','Italy','Greece'


# In[107]:


df2usa.head(100)


# In[108]:


sns.catplot(x="location", y="new_cases", kind="swarm", hue="continent", data=df2usa)


# **Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook**

# In[128]:


sns.set(font_scale=2)
ax = sns.violinplot(x="location", y="new_cases", data=df2usa, inner=None)
ax.set_xlabel("X-Axis", fontsize = 20)
ax.set_ylabel("Y-Axis", fontsize = 20)

ax = sns.swarmplot(x="location", y="new_cases", data=df2usa,
                   color="white", edgecolor="gray")
ax.set_title("Distribution of new covid19 casese between 2021-01-01 and 2021-02-01", fontsize=20)


# **Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.**

# 
# The data was pulled from Kaggle website and the dataset has a daily record new covid cases for many countries. During my data preparation have filtered out those countries that was requested based on the question and also filtered out for the specified timeframe. 
# The graph clearly addressed the question asked. The graph has shown the distribution of new covid cases. As indicated on the graph, the x-axis is the location indicating the countries requested and the y-axis is the count that the daily count of the new covid cases. 
# The graph's title and 'Y' and 'X' labels are bold and clearly help the users to better to focus on the objective needed for plotting the graph. The color used for the violin plot and the white dots in the middle are based on the gestalt principles. The white dots with different background also draw the attention of the users.
# From the graph it is possible to see that the new covid cases in South Africa started earlier, within the timeframe give, as compared to Mexico and Italy.
# The graph is a violin and swarm combination plot. 
# 
# 

# Task 1: Develop one question related to geospatial aspects of this dataset. Write the question in a Markdown cell in your notebook.
# 
# Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.
# 
# Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.
# 
# Task 4: DO NOT SUBMIT YOUR NOTEBOOK FOR THIS. Hover over the graphic in your notebook and you'll see a tiny save icon. Click on that and save the graphic as an image. Upload the image, write your question and answer in the assignment submission box, and submit everything.
# 
# Reviewer (max 2 points, one per review): Open the graphic. Assign points using the criteria provided. Also provide feedback if you wish. Submit.

# **Task 1: Develop one question related to geospatial aspects of this dataset. Write the question in a Markdown cell in your notebook.**

# **What is the distribution of new covid cases for few selected countries in the year 2020?**

# **Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.**
# 

# In[309]:


print(df2)


# In[6]:


df4 = df2[(df2.index > '2020-01-01') & (df2.index < '2020-12-31') & (df2['location'].isin(['Brazil','China','Canada','Italy', 'Russia','United Kingdom','Ethiopia','Ghana','United States','Germany','Argentina','Australia']))]


# In[65]:


df5=df4[['new_cases','location']].groupby([pd.Grouper(freq="A"), "location"]).sum()


# In[66]:


df5.reset_index(inplace = True)


# In[67]:


df5.rename(columns = {'location':'country'}, inplace = True)


# In[68]:


print(df5)


# In[5]:



np.random.seed(12)
gapminder = px.data.gapminder().query("year==2007")
#gapminder['counts'] = np.nan



df=pd.merge(gapminder, df5[['country', 'new_cases']], how='left', on='country')

fig = px.choropleth(df, locations="iso_alpha",
                    color="new_cases",   
                    hover_name="country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(
    title_text = 'Count distribution of new Covid19 cases for 2020',
  
)

fig.show()


# **Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.**
# 

# Visualizing concepts and qualitative data
# Use the dataset and Jupyter Notebook you've been using. 
# Submitter (max 8 points): 
# Task 1: Develop one question related to concepts or qualitative aspects of this dataset. Write the question in a Markdown cell in your notebook.
# Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.
# Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.
# Task 4: DO NOT SUBMIT YOUR NOTEBOOK FOR THIS. Hover over the graphic in your notebook and you'll see a tiny save icon. Click on that and save the graphic as an image. Upload the image, write your question and answer in the assignment submission box, and submit everything.
# Reviewer (

# **Task 1: Develop one question related to concepts or qualitative aspects of this dataset. Write the question in a Markdown cell in your notebook.**
# 
# **What are the most mentioned characteristics about wine and which one is the most popular wine from the give dataset?**
# 

# In[22]:


dfwine.head()


# In[26]:


# The first five rows of the wind dataset 

dfwine.info()


# In[26]:


dfwine.head()


# In[ ]:


dfwine.


# **Task 2: Develop a graphic appropriate to this type of question, that answers the question. Use a code cell in your notebook.**

# In[22]:


# Start with one review:
text = dfwine.description[325]

# Create and generate a wo3d cloud image:
wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[19]:


# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[27]:


text = " ".join(review for review in dfwine.description)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[34]:


# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')

plt.title("Wine expert's testing notes on different brand of wines")
plt.axis("off")
plt.show()


# **Task 3: Write your interpretation of the graph (100 to 500 words), including how it answers the question.**

# I have pulled a dataset from Kaggle.com. The dataset name is winemag-data-130k-v2 and it is a csv file. I have installed wordcloud and uploaded the necessary python libraires including: WordCloud, STOPWORDS, and ImageColorGenerator. There are 129,970 records and 13 columns. I have used a column called “description” where there are some details about the specific wine testing. The dataset has information about the country, variety and winery about the wine. After applying the python code, I was able to parse all the words in the description column and I have found about 31 million words and I have applied the WordCloud to generate the image. 
# 
# The image basically shows the most commonly used words in the description column about the wine. The image generated have addressed the question that I was looking for. The most mentioned characteristics about the wine are: finish, palate, aroma, nose and black cherry. It looks like black cheery, full bodied and rich have been mentioned also relatively to certain large extent. This may be an indication that maybe black cheery wine are most popular in certain country. This might need further analysis. 
# 
# Potentially, more findings from the notes/description can be obtained, I have only applied limited application of WordCloud libraires on my code in order to address the question asked.   
#    
# 

# In[ ]:




