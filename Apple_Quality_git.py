#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import random
from sklearn.cluster import KMeans
from scipy import stats


# In[2]:


#importing the data set
inpPath = "/Users/linus/Documents/CA259/02_assignement_two/"
apples = pd.read_csv(inpPath + "apple_quality.csv", delimiter =  ",", header = 0)
apples


# In[3]:


#dropping the last row because it just contains the author of the data set
apples = apples.drop(4000)
apples


# In[4]:


#dropping the A_id column which holds the ID for each apple
apples.drop("A_id", axis=1, inplace = True)
apples


# In[5]:


#making the describe table more pretty to use it in the slides
apples.describe().style.background_gradient(axis=1, cmap='Blues')


# In[6]:


# because Acidity does not show up in the table, need to check the type
type(apples.iloc[0,6])


# In[7]:


#change type to float
apples["Acidity"] = apples["Acidity"].astype("float")
type(apples.iloc[0,6])


# In[8]:


#getting the describe table again
apples.describe().style.background_gradient(axis=1, cmap='Blues')


# In[9]:


#checking for missing values
apples.isna().sum()


# In[11]:


#the following cells calculate skew and kurtosis for every feature

print(apples["Size"].skew())
print(apples["Size"].kurtosis())


# In[13]:


print(apples["Weight"].skew())
print(apples["Weight"].kurtosis())


# In[15]:


print(apples["Sweetness"].skew())
print(apples["Sweetness"].kurtosis())


# In[17]:


print(apples["Crunchiness"].skew())
print(apples["Crunchiness"].kurtosis())


# In[19]:


print(apples["Juiciness"].skew())
print(apples["Juiciness"].kurtosis())


# In[21]:


print(apples["Ripeness"].skew())
print(apples["Ripeness"].kurtosis())


# In[23]:


print(apples["Acidity"].skew())
print(apples["Acidity"].kurtosis())


# In[24]:


#getting an overview over all histograms at the same time for the numerical columns
#putting all numerical columns in a variable to iterate over
numerical_columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

#specifying the color and the size
plt.figure(figsize=(15, 10))
sns.set_palette("tab10")

#iterating over the numerical columns to plot
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=apples, x=column, kde=True, bins=20)  
    plt.title(column)

plt.tight_layout()
plt.show()


# In[25]:


#checking if there are really only two values for apple quality
apples["Quality"].unique()


# In[26]:


#graphical representation about the quality distribution
quality_counts = apples["Quality"].value_counts()

plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', textprops={'fontsize':10})
plt.title('Quality Distribution')
plt.show()


# In[27]:


#encoding the quality variable as numerical (0 or 1) for better use
encoded_dict = {'good': 1, 'bad': 0}
apples["Quality_numeric"] = apples["Quality"].map(encoded_dict)
apples


# In[28]:


#Box whisper plots by quality to better visualise the difference between good and bad apples for features
plt.figure(figsize=(15, 10))
sns.set_palette("Set2")

for i, column in enumerate(apples.columns[:-2]):  
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='Quality_numeric', y=column, data=apples)
    plt.title(f'{column} by Quality_numeric')

plt.tight_layout()
plt.show()


# In[30]:


#creating a correlation matrix to get a better idea for connections between variables
apples_corr = apples.drop('Quality', axis=1).corr()
apples_corr


# In[31]:


'''sns.heatmap(apples_corr, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()'''


# In[32]:


#creating a correlation heatmap with no correlation duplicates by employing a mask

mask = np.triu(np.ones_like(apples_corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(apples_corr, mask=mask, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# In[33]:


#Analysing the correlations pairs with a correlation higher or lower and equal to 0.2 and -0.2
tpllst = []
for i in range(0,8):
    for j in range(0,8):
        if apples_corr.iloc[i,j]>=0.2 and apples_corr.iloc[i,j]<1: 
            tpllst.append((apples_corr.index[i], apples_corr.index[j], apples_corr.iloc[i,j]))
        elif apples_corr.iloc[i,j]<= -0.2 and apples_corr.iloc[i,j]>-1:
            tpllst.append((apples_corr.index[i], apples_corr.index[j], apples_corr.iloc[i,j]))

dltlst = [] #new list to save only those items where the quality_numeric is not at the second place underlining that this variable makes no sense as independent variable
for i in range(len(tpllst)):
    if tpllst[i][1] != "Quality_numeric":
        dltlst.append(tpllst[i])
dltlst


# In[34]:


#trying to find out which features can be predicted by other features
#linear regressions for variables with 2 or more correlations above/equal 0.2 or below/equal -0.2

# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Size','Ripeness']]
yDf = apples['Sweetness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_sweet = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_sweet.score(X_test, y_test))
print(reg_lin_sweet.coef_)



# In[35]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Juiciness','Ripeness']]
yDf = apples['Crunchiness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_crunch = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_crunch.score(X_test, y_test))
print(reg_lin_crunch.coef_)


# In[36]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Weight','Sweetness', 'Crunchiness', 'Acidity']]
yDf = apples['Ripeness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_ripe = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_ripe.score(X_test, y_test))
print(reg_lin_ripe.coef_)


# In[37]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Juiciness','Ripeness']]
yDf = apples['Acidity']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_ripe = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_ripe.score(X_test, y_test))
print(reg_lin_ripe.coef_)


# In[38]:


#predicting the quality with a logistic regression (logit because we look at binary feature)
# using only a selection of features based on the correlations list
xDf = apples[['Size','Sweetness','Juiciness','Ripeness']] #only the values with the highes
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)


# Evaluate the accuracy of the model
print(logreg.score(X_test, y_test))


# In[40]:


#trying the regression again with only Size as the indepent variable
xDf = apples[['Size']]
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)


# Evaluate the accuracy of the model
print(logreg.score(X_test, y_test))


# In[71]:


#Besides just including 4 independent variables, all features were included leading to a even higher significance
xDf = apples[['Size','Sweetness','Juiciness','Ripeness', 'Weight', 'Acidity', 'Crunchiness']]
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg2 = LogisticRegression()

# Fit the model to the training data
logreg2.fit(X_train, y_train)

print(logreg2.score(X_test, y_test))


# In[42]:


#trying to cluster the apples into different variates

#setting random seet for reproduceability 
random.seed(42)

#dropping the non numerical Quality column
xDf = apples.drop(columns='Quality')
#correct import


#initialize inertia list and get the k and the inertia per k (or WCSS per k)
inertiaLst = []
for kVal in range(1, 10):
    kmeans = KMeans(n_clusters=kVal)
    kmeans.fit(xDf)
    inertiaLst.append([kVal, kmeans.inertia_])

#transposing and plotting the inertia list
inertiaArr = np.array(inertiaLst).transpose()
plt.plot(inertiaArr[0], inertiaArr[1])
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


# In[43]:


#building the k-means model
#read kVal at the "elbow"
kVal = 3

#set the k for the KMeans to kVal
kmeans = KMeans(n_clusters=kVal)

#fit the model
kmeans.fit(xDf)

#get the label from the kmeans and add it as a column to the apples table
apples['label'] = kmeans.labels_


# In[44]:


apples


# In[45]:


#getting a visual for the label distribution
line_colors = ['blue','green', 'gray'] 
label_counts = apples["label"].value_counts()

plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', textprops={'fontsize':10}, colors = line_colors)
plt.title('Label Distribution')
plt.show()


# In[46]:


#getting the exact number of apples per variety
label_counts


# In[47]:


#exploring the differences between the clusters by examening the mean for the features 
apples.drop(columns = "Quality").groupby(by = "label").mean()


# In[48]:


#getting the centroids for the clusters of the k means algorithm
#aim is to build a plot showing the mean for each feature per cluster
centroids = kmeans.cluster_centers_
print(centroids)


# In[49]:


#creating a table with the mean of the features per cluster
centroids = kmeans.cluster_centers_
feature_names = ['Size', 'Weight', 'Sweetness' , 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity' ,'Quality_numeric']

# Create a DataFrame to store the centroids and feature names
centroid_table = pd.DataFrame(centroids, columns=feature_names)

centroid_table


# In[50]:


#transposing the table
transposed_centroid = centroid_table.transpose()
transposed_centroid


# In[51]:


#turning the table into a graph to better see the feature profile of the apple variaties 
line_colors = ['gray', 'green', 'blue'] 
fig = plt.figure(figsize=(12, 8))
for i, column in enumerate(transposed_centroid.columns):
    plt.plot(transposed_centroid[column], marker='o', linestyle='', markersize=12, color=line_colors[i])
plt.title('Characteristics for each Apple Variety', fontweight='bold')
plt.xlabel("Features", fontweight='bold')
plt.ylabel("Centroid Value", fontweight='bold')
plt.legend(transposed_centroid.columns)


# In[52]:


#building several join plots below to show the differences between the clusters in certail features
plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Juiciness', hue='label', data=apples, palette=line_colors, s=9)


# In[53]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Ripeness', hue='label', data=apples, palette=line_colors, s=9)


# In[54]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)


# In[55]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Juiciness', y='Ripeness', hue='label', data=apples, palette=line_colors, s=9)


# In[56]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Juiciness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)


# In[57]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Ripeness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)



# In[58]:


#to later conduct t tests, three new dataframes, one per cluster will need to be build
apples_0 = apples[apples["label"] == 0]
apples_0


# In[59]:


apples_1 = apples[apples["label"] == 1]
apples_1


# In[60]:


apples_2 = apples[apples["label"] == 2]
apples_2


# In[65]:


#Seeing the differences in the mean, t tests were employed to test if these differences occur by chance or not


#instead of manually entering the clusters max and min values this code was created to address the issue that the cluster names 
#might be different.

#this get the highest and the lowest cluster name per feature
sweet_max = centroid_table['Sweetness'].idxmax()
sweet_min = centroid_table['Sweetness'].idxmin()

#checking the cluster which has the max value per feature and setting the df accordingly to get the values only per feature
if sweet_max == 0:
    sweet_max_df = apples_0['Sweetness']
elif sweet_max == 1:
    sweet_max_df = apples_1['Sweetness']
elif sweet_max == 2:
    sweet_max_df = apples_2['Sweetness']
    
#same for the minimal value
if sweet_min == 0:
    sweet_min_df = apples_0['Sweetness']
elif sweet_min == 1:
    sweet_min_df = apples_1['Sweetness']
elif sweet_min == 2:
    sweet_min_df = apples_2['Sweetness']

# Perform independent t-test with cluster with max value for feature against cluster with min value per feature
t_statistic, p_value = stats.ttest_ind(sweet_max_df, sweet_min_df)

# Print the results
print(f'Test: Is Apple Variaty {sweet_max} systematically more sweet than Apple Variaty {sweet_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)



# In[66]:


#the same for the other features follows below
juicy_max = centroid_table['Juiciness'].idxmax()
juicy_min = centroid_table['Juiciness'].idxmin()

if juicy_max == 0:
    juicy_max_df = apples_0['Juiciness']
elif juicy_max == 1:
    juicy_max_df = apples_1['Juiciness']
elif juicy_max == 2:
    juicy_max_df = apples_2['Juiciness']
    

if juicy_min == 0:
    juicy_min_df = apples_0['Juiciness']
elif juicy_min == 1:
    juicy_min_df = apples_1['Juiciness']
elif juicy_min == 2:
    juicy_min_df = apples_2['Juiciness']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(juicy_max_df, juicy_min_df)

# Print the results
print(f'Test: Is Apple Variaty {juicy_max} systematically more juicy than Apple Variaty {juicy_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[67]:


ripe_max = centroid_table['Ripeness'].idxmax()
ripe_min = centroid_table['Ripeness'].idxmin()

if ripe_max == 0:
    ripe_max_df = apples_0['Ripeness']
elif ripe_max == 1:
    ripe_max_df = apples_1['Ripeness']
elif ripe_max == 2:
    ripe_max_df = apples_2['Ripeness']
    

if ripe_min == 0:
    ripe_min_df = apples_0['Ripeness']
elif ripe_min == 1:
    ripe_min_df = apples_1['Ripeness']
elif ripe_min == 2:
    ripe_min_df = apples_2['Ripeness']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(ripe_max_df, ripe_min_df)

# Print the results
print(f'Test: Is Apple Variaty {ripe_max} systematically more ripe than Apple Variaty {ripe_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[68]:


acid_max = centroid_table['Acidity'].idxmax()
acid_min = centroid_table['Acidity'].idxmin()

if acid_max == 0:
    acid_max_df = apples_0['Acidity']
elif acid_max == 1:
    acid_max_df = apples_1['Acidity']
elif acid_max == 2:
    acid_max_df = apples_2['Acidity']
    

if acid_min == 0:
    acid_min_df = apples_0['Acidity']
elif acid_min == 1:
    acid_min_df = apples_1['Acidity']
elif acid_min == 2:
    acid_min_df = apples_2['Acidity']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(acid_max_df, acid_min_df)

# Print the results
print(f'Test: Have Apples from Variaty {acid_max} systematically more Acidity than Apples from Variaty {acid_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)

