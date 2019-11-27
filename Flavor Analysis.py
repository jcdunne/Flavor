#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import Data
data = pd.read_excel('Final Flavor Data.xlsx', sheet_name='Data')


# In[ ]:


# Head of Data
#data.head()


# In[ ]:


# Calculate variance for subsetting Data
from statistics import variance

for col in data.columns[4:]:
    print("Variance of %s = %s"%(col, variance(data[col].dropna())))


# In[ ]:


# Descriptive statistics of columns 5 through 12
data[list(data.columns[4:17])].dropna().describe()


# In[ ]:


# Columns 13 through 17
#sns.pairplot(data[list(data.columns[4:17])].dropna())


# In[ ]:


# Data columns for subsetting - Columns 1 through 17 dropping Astringent
data.columns


# In[ ]:


# Subset of Data
data_sub = data[['GIN', 'NC_Accession', 'Seed Source', 'Rep', 'mean_oil', 'raw_mc_ww',
       'roast_color', 'paste_color', 'dark_roast', 'raw_bean', 'roast_peanut',
       'sweet_aromatic', 'sweet', 'bitter', 'wood_hulls_skins','cardboard']].dropna()


# In[ ]:


# Pairplot for subset of Data
#sns.pairplot(data_sub[list(data_sub.columns[4:])])


# In[ ]:


# Checks?
Cultivars = ['Bailey','Bailey II','Emery','Sullivan','Wynne','Georgia 06-G' ]


# In[ ]:


data_sub['Check'] = data_sub['NC_Accession'].apply(lambda x: 'Yes' if x in Cultivars else 'No')


# In[ ]:


# Creates new columsn for 2 - 10 clusters to view in interactive plots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Control Panel
clus_total = 10
pca_comp = 5

for i in range(2,(clus_total+1)):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_sub.iloc[:,4:16])
    
    data_sub['Clusters_%s'%(i)] = kmeans.labels_
    
scaler = StandardScaler()
scaler.fit(data_sub.iloc[:,4:16])
scaled_data = scaler.transform(data_sub.iloc[:,4:16])

pca = PCA(n_components=pca_comp)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

for i in range(0,pca_comp):
    data_sub['PCA%s'%(i+1)] = x_pca[:,i]


# In[ ]:


# Interactive Plots to Visualice Clusters - Use the Control Panel to See the number of clusters, pca_init and pca_post data
import plotly.express as px
# Control Panel

# Select number of clusters (2-10)
Clusters = 3
# Select PCA component (1-5)
PCA_init = 'PCA1'
# Select PCA component (1-5)
PCA_post = 'PCA2'

# Create a new data column, 'Cultivars' and 'Cluster i' where i is the number of clusters defined in Clusters
def new_char(cols):
    Check = cols[0]
    Cluster = cols[1]
    
    if Check == 'Yes':
        return 'Cultivar'
    
    else:
        for i in range(0,Clusters):
            if Cluster == i:
                return 'Cluster %s'%(i+1)

# Set the new column to color for dispaying the colors below
Color = data_sub[['Check','Clusters_%s'%(Clusters)]].apply(new_char, axis=1)
# Plot using plotly.express
fig = px.scatter(data_sub, x="%s"%(PCA_init), y="%s"%(PCA_post), opacity=0.7, color=Color ,hover_data=['NC_Accession'], template='plotly_white')
# Change the marker sizes and attributes
fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
# Display the figure
fig.show()

#Culitvar = Bailey, Georgia 06-G

