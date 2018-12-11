
# coding: utf-8

# # SNAP Recipiency in the 2017 ACS Survey 
# 
# ** This project was conducted for Northwest Harvest, Washington state's main hunger relief agency. It aims to understand where hungry people in 2017 reside in WA state, as well as who they are to help Northwest Harvest better target their programming efforts. The main data come from the 2017 American Community Survey. **

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
import fiona
# If below doesn't work, run in cmd: conda install -c conda-forge basemap 
from mpl_toolkits.basemap import Basemap
from descartes import PolygonPatch
import os
import seaborn as sns


# ### Relevant 2017 variables

# **Housing Record**
# 
# Dep. Var 
# - FS (Yearly food stamp/SNAP recipiency)
# 
# Ind. Var
# - PUMA
# - WGTP (household weight) 
# - ADJINC (adjustment factor for income)
# - HINCP (mean household income)
# - NP (Number of persons in this household)
# - HHT (Household/family type) (Note: Could determine proportion of partnered/unpartnered households with children by crosstabbing with transformed HUPAC)
# - FES (Family type and employment status)
# - HHL (primary language spoken)
# - HUPAC (presence and age of children) (Note: need variable transformation to a binary and drop NA's depending on whether the age is important) 
# 
# **Person Record**
# 
# Ind. Var 
# - PUMA
# - PWGTP (person's weight) 
# - ADJINC (adjustment factor for income) 
# - WAGP (wages/salary income the past 12 mo.) 
# - AGE 
# - ENG (ability to speak English) 
# - MIL (military service) 
# - SEX 
# - HICOV (health insurance coverage) 
# - NATIVITY 
# - RAC1P (race)
# - ESP (employment status of parents)
# - JWTR (means of transportation to work)
# - OCCP (occupation) (Note: Top 5-10 industries of employment, for example food industry, construction, etc. VS. job within industry. It would be handy to have the proportion in the industries as well)

# ### Merge Person to Household datasets

# Connecting HH and Person records (scroll to bottom "Merging Person and Housing Unit Files": 
# https://www.census.gov/programs-surveys/acs/technical-documentation/pums/filestructure.html
# 
# HH Record DD: 
# RT = Record Type 
# SERIALNO = Housing unit/GQ person serial number 
# 
# Person Record DD: 
# RT = Record Type
# SERIALNO = Housing Unit/GC person serial number
# SPORDER = Person number (1-20)
# 
# Downloaded person data from: 
# https://www2.census.gov/programs-surveys/acs/data/pums/2016/1-Year/

# In[5]:

# Load household-level data
acs_17_hh = pd.read_csv('Data/psam_h53.csv')


# In[6]:

len(acs_17_hh)


# In[7]:

# Number of housing units in Washington state
acs_17_hh.WGTP.sum()


# In[8]:

acs_17_hh.head()


# In[9]:

# Load person-level data
acs_17_p = pd.read_csv('Data/psam_p53.csv')


# In[10]:

acs_17_p.head()


# In[11]:

acs_17_p = acs_17_p[acs_17_p.SPORDER==1]


# Note everything we conclude at the person level is about household HEADS at this point on. 

# In[12]:

acs_17_hh.sort_values(by="SERIALNO",inplace=True)


# In[13]:

acs_17_p.sort_values(by="SERIALNO",inplace=True)


# In[14]:

pumas_hh_merge = acs_17_hh.merge(acs_17_p,how="left",on=["SERIALNO", "PUMA", "ST"])


# In[15]:

pumas_hh_merge.head()


# In[16]:

pumas_hh_merge.reset_index(inplace=True)


# ### Make basic adjustments

# In[17]:

# adjust household income
pumas_hh_merge.HINCP = pumas_hh_merge.HINCP * (pumas_hh_merge.ADJINC_x/1000000)


# In[18]:

# adjust person's wages 
pumas_hh_merge.WAGP = pumas_hh_merge.WAGP * (pumas_hh_merge.ADJINC_x/1000000)


# In[19]:

# explore a vacant household and make sure person-level data is NA / missing
pumas_hh_merge[pumas_hh_merge.NP==0]['SEX'].head()


# In[20]:

# drop vacant households
print(len(pumas_hh_merge))
print(len(pumas_hh_merge.dropna(subset=['NP','HINCP'])))
pumas_hh_merge = pumas_hh_merge.dropna(subset=['NP','HINCP'])


# In[21]:

# Confirm no null values because vacant households were excluded 
print(pumas_hh_merge['FS'].isnull().values.any())
print(pumas_hh_merge['FS'].isnull().sum())


# ### Cross-tab main dependent variable, FS 

# FS = Yearly food stamp/Supplemental Nutrition Assistance Program (SNAP) recipiency
# 
# b .N/A (vacant)
# 
# 1 .Yes
# 
# 2 .No

# In[22]:

pumas_hh_merge.groupby('FS').sum()['WGTP']
# Automatically excludes group quarters because their WGTP = 0


# In[24]:

# Crosstab of # of households on SNAP & not on SNAP by PUMA
pd.crosstab(pumas_hh_merge.PUMA,pumas_hh_merge.FS,pumas_hh_merge.WGTP,aggfunc=sum,margins=True).head()


# In[25]:

df = pd.crosstab(pumas_hh_merge.PUMA,pumas_hh_merge.FS,pumas_hh_merge.WGTP,aggfunc=sum,margins=True)


# In[26]:

column_names = df.columns.values
column_names[0] = 'receives_snap'
column_names[1] = 'no_snap'
df.columns = column_names
df.reset_index(inplace=True)
df['perc_snap'] = df.receives_snap/df.All


# In[27]:

df.head()


# In[29]:

# Note "All" is a string type while all other "PUMA" values are int
print(type(df.PUMA[55]))
print(type(df.PUMA[len(df.PUMA)-1]))


# ### Plot location of percentages

# In[30]:

# Load PUMA shapefile for WA state
puma_shape = fiona.open("Data/cb_2017_53_puma10_500k.shp")


# In[31]:

first = puma_shape.next()
first['properties']['PUMACE10']


# In[32]:

# See what single element in shapefile looks like
print(puma_shape.schema)


# In[33]:

# Create basemap 
all_polygons = [item[1] for item in puma_shape.items()]


# In[34]:

fig = plt.figure()
ax = fig.gca()
BLUE = '#6699cc'
for poly in all_polygons:
    ax.add_patch(PolygonPatch(poly['geometry'], fc=BLUE, ec=BLUE, alpha=0.5,zorder=2))
ax.axis('scaled')
plt.show()


# In[35]:

# Write percent snap data to geographic file
    # Only have to run this once. Just load "intersection.shp" after first time (go next cell onwards)
# Reference this page: https://gis.stackexchange.com/questions/178765/intersecting-two-shapefiles-from-python-or-command-line

schema =  {'geometry': 'Polygon','properties': {'puma_code': 'int','perc_snap':'float'}}

with fiona.open('intersection.shp', 'w',driver='ESRI Shapefile', schema=schema) as output:
    for pum in fiona.open('Data/cb_2017_53_puma10_500k.shp'):
        for index, row in df.iterrows():
            if int(pum['properties']['PUMACE10']) == row.PUMA:
                #print(row.PUMA)
                prop = {'puma_code': row.PUMA, 'perc_snap': row.perc_snap}
                output.write({'geometry':mapping(shape(pum['geometry'])),'properties': prop})
                #print('mapping worked!')


# In[37]:

merged_shape = fiona.open("intersection.shp")


# In[38]:

df


# In[39]:

# Map shapefile for WA state
fig = plt.figure(figsize=(20,10))
ax = fig.gca()
alphas = []

for _, poly in merged_shape.items():
    ax.add_patch(PolygonPatch(poly['geometry'], fc='BLUE', ec='BLUE', alpha=poly['properties']['perc_snap']+0.15))    
    alphas.append(poly['properties']['perc_snap'])
ax.axis('scaled')
ax.grid(False)
plt.figtext(.5,.84,'Perc. of Households receiving SNAP, WA State', fontsize=25, fontweight='bold',ha='center')
#plt.figtext(.5,.85,'(Darkest: '+str(int(max(alphas)*100))+ '%   Lightest: '+str(int(min(alphas)*100))+'%)', fontsize=20, ha='center')
#plt.figtext(.5,.85,'(Dark = High)', fontsize=20, ha='center')

# Label percentages 
plt.text(-123.5,47.7,'11%',fontsize=13) #11900
plt.text(-123.7,47.2,'21%',fontsize=13) #11300
plt.text(-123.8,46.5,'20%',fontsize=13) #11200
plt.text(-122.3,46.5,'15%',fontsize=13) #11000
plt.text(-122,48.8,'14%',fontsize=13) #10100
plt.text(-122,48.5,'12%',fontsize=13) #10200
plt.text(-122,48.2,'10%',fontsize=13) #11706
plt.text(-122,47.9,'8%',fontsize=13) #11705
plt.text(-121.75,47.6,'1%',fontsize=13) #11616
plt.text(-121.9,47.25,'6%',fontsize=13) #11615
plt.text(-122.2,47,'12%',fontsize=13) #11507
plt.text(-120.2,48.5,'17%',fontsize=13) #10400
plt.text(-120.8,47.7,'9%',fontsize=13) #10300
plt.text(-120.5,47.1,'17%',fontsize=13) #10800
plt.text(-120.6,46.6,'24%',fontsize=13) #10901
plt.text(-121,46.4,'28%',fontsize=13) #10902
plt.text(-117.5,47.72,'19%',fontsize=13) #10501
plt.text(-117.55,47.6,'16%',fontsize=13) #10502
plt.text(-117.2,47.6,'16%',fontsize=13) #10503
plt.text(-117.7,47.4,'9%',fontsize=13) #10504
plt.text(-118.4,47,'12%',fontsize=13) #10600
plt.text(-119.5,46.3,'16%',fontsize=13) #10701
plt.text(-119.25,46.1,'13%',fontsize=13) #10702
plt.text(-119,46.5,'14%',fontsize=13) #10703
plt.text(-122.7,46.85,'9%',fontsize=13) #11402
plt.text(-122.9,47,'11%',fontsize=13) #11401
plt.text(-122.8,45.56,'20%',fontsize=13) #11101
plt.text(-122.7,45.7,'9%',fontsize=13) #11102
plt.text(-122.4,45.55,'8%',fontsize=13) #11103
plt.text(-122.5,45.75,'13%',fontsize=13) #11104
plt.text(-122.7,47.7,'9%',fontsize=13) #11801
plt.text(-122.75,47.5,'14%',fontsize=13) #11802
plt.text(-122.8,47.25,'6%',fontsize=13) #11502

plt.axis('off')
plt.savefig('WA_hunger_location.png')
plt.show()


# In[40]:

seattle_pumas = [11601, 11602, 11603, 11604, 11605, 11606, 11610, 11611]


# In[41]:

# Map shapefile for City of Seattle 
fig = plt.figure(figsize=(10,12))
ax = fig.gca()
alphas = []
for _, poly in merged_shape.items():
    if poly['properties']['puma_code'] in seattle_pumas:
        ax.add_patch(PolygonPatch(poly['geometry'], fc='BLUE', ec='BLUE', alpha=poly['properties']['perc_snap']+0.15,zorder=2))
        alphas.append(poly['properties']['perc_snap'])
ax.axis('scaled')
ax.grid(False)
plt.figtext(.5,.86,'Perc. of Households receiving SNAP, City of Seattle', fontsize=25, fontweight='bold',ha='center')
#plt.figtext(.5,.85,'(Darkest: '+str(int(max(alphas)*100))+ '%   Lightest: '+str(int(min(alphas)*100))+'%)', fontsize=20, ha='center')
#plt.title('Percentage of Households receiving SNAP (dark = high)')

# Label percentages
plt.text(-122.38,47.68,'5%',fontsize=13) #11601
plt.text(-122.30,47.68,'6%',fontsize=13) #11602
plt.text(-122.40,47.64,'6%',fontsize=13) #11603
plt.text(-122.31,47.61,'12%',fontsize=13) #11604
plt.text(-122.38,47.55,'13%',fontsize=13) #11605
plt.text(-122.34,47.75,'6%',fontsize=13) #11606
plt.text(-122.25,47.488,'14%',fontsize=13) #11610
plt.text(-122.33,47.48,'17%',fontsize=13) #11611

plt.axis('off')
plt.savefig('Seattle_hunger_location.png')
plt.show()


# ## Start Interesting Statistics
# Reference data dictionary to understand labels: https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2017.pdf

# ### Baseline

# In[42]:

df.tail()


# In[43]:

df.to_csv(path_or_buf='SNAP_by_PUMA.csv')


# In[44]:

df.loc[56]
# 12% of WA households receive SNAP


# ### Household Income

# In[47]:

df = pd.crosstab(pumas_hh_merge.HINCP,pumas_hh_merge.FS,pumas_hh_merge.WGTP,aggfunc=sum,margins=True)
column_names = df.columns.values
column_names[0] = 'receives_snap'
column_names[1] = 'no_snap'
df.columns = column_names
df.reset_index(inplace=True)
df['% of SNAP recipients'] = df.receives_snap/(df.receives_snap.values[-1])*100
df['% of total population'] = df.All/(df.All.values[-1])*100
df = df.drop(['receives_snap','no_snap','All'],axis=1)
df = df[:-1]


# Make plots
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(df['HINCP'],df['% of SNAP recipients'],'k')
plt.ylabel('% of Total Population',fontsize=16)
plt.xlim((None,500000))
plt.title('Household Income, Total Population',fontsize=20)
plt.subplot(2,1,2)
plt.plot(df['HINCP'],df['% of total population'],'k')
plt.ylabel('% of SNAP Population',fontsize=16)
plt.xlabel('Income ($)',fontsize=16)
plt.xlim((None,500000))
plt.title('Household Income, SNAP Population',fontsize=20)
plt.savefig('Household_Income.png')
plt.show()


# ### Age of Household Head

# In[48]:

# Histograms, frequency of each age group
df = pd.crosstab(pumas_hh_merge.AGEP,pumas_hh_merge.FS,pumas_hh_merge.WGTP,aggfunc=sum,margins=True)
column_names = df.columns.values
column_names[0] = 'receives_snap'
column_names[1] = 'no_snap'
df.columns = column_names
df.reset_index(inplace=True)
df['% of SNAP recipients'] = df.receives_snap/(df.receives_snap.values[-1])*100
df['% of total population'] = df.All/(df.All.values[-1])*100
df = df[:-1]
df['AGEP'] = df['AGEP'].astype('int')
print(df.head())


# In[49]:

df.fillna(0,inplace=True)


# In[50]:

# Make plots
    # Source for xticks: https://stackoverflow.com/questions/27083051/matplotlib-xticks-not-lining-up-with-histogram
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.hist(df.AGEP,weights=df.All,bins=np.arange(0,max(df.AGEP)+4,1)-0.5,color='blue',edgecolor='black',alpha=0.3) # 1 year
plt.xticks(np.arange(min(df.AGEP), max(df.AGEP)+4,2))
plt.xlim([14,100])
plt.ylabel('# of Household Heads',fontsize=16)
plt.title('Histogram of Age of Household Head, Total Population',fontsize=20)
plt.hist(df.AGEP,weights=df.receives_snap,bins=np.arange(0,max(df.AGEP)+4,1)-0.5,color='red',edgecolor='black',alpha=0.3) # 1 year
plt.xticks(np.arange(min(df.AGEP), max(df.AGEP)+4,2))
plt.ylabel('# of Household Heads',fontsize=16)
plt.xlabel('Age (Years)',fontsize=16)
plt.xlim([14,100])
plt.title('Histogram of Age of Household Head',fontsize=20)

#create legend
handles = [Rectangle((0,0),1,1,color=c,ec="k",alpha=0.5) for c in ['blue','red']]
labels= ["Total Population","SNAP population"]
plt.legend(handles, labels,fontsize=16)

plt.savefig('outputs/Hist_Household_Head_Age.png')
plt.show()


# ### Parameterized tables

# **For categorical variables, SNAP v. non-SNAP** 

# In[51]:

def build_table_cat_sns(pumas_table,var,weight):
    df = pd.crosstab(pumas_table[var],pumas_table.FS,pumas_table[weight],aggfunc=sum,margins=True)
    column_names = df.columns.values
    column_names[0] = 'receives_snap'
    column_names[1] = 'no_snap'
    df.columns = column_names
    df.reset_index(inplace=True)
    df['% Receiving SNAP'] = df.receives_snap/df.All*100
    df['% Not Receiving SNAP'] = df.no_snap/df.All*100
    df['Total Population'] = df.All/df.All*100
    df = df.drop(['receives_snap','no_snap','All'],axis=1)
    for index, row in df.iterrows():
        df.at[index, '% Receiving SNAP'] = "%.2f" %row['% Receiving SNAP']
        df.at[index, '% Not Receiving SNAP'] = "%.2f" %row['% Not Receiving SNAP']
        df.at[index, 'Total Population'] = "%.2f" %row['Total Population']
    print(df)
    return df 


# **For categorical variables, SNAP v. Total Population** 

# In[52]:

def build_table_cat_stot(pumas_table,var,weight):
    df = pd.crosstab(pumas_table[var],pumas_table.FS,pumas_table.WGTP,aggfunc=sum,margins=True)
    column_names = df.columns.values
    column_names[0] = 'receives_snap'
    column_names[1] = 'no_snap'
    df.columns = column_names
    df.reset_index(inplace=True)
    #print(df)
    df['% of SNAP recipients'] = df.receives_snap/(df.receives_snap.values[-1])*100
    df['% of total population'] = df.All/(df.All.values[-1])*100
    df = df.drop(['receives_snap','no_snap','All'],axis=1)
    for index, row in df.iterrows():
        df.at[index, '% of SNAP recipients'] = "%.2f" %row['% of SNAP recipients']
        df.at[index, '% of total population'] = "%.2f" %row['% of total population']
    print(df)
    return(df)


# In general, SNAP v. Total Population is more intuitive than SNAP v. non-SNAP; thus it is used more often below (e.g. for generating plots and graphs) 

# ### Language spoken at home

# In[55]:

df_hhl_sns = build_table_cat_sns(pumas_hh_merge,'HHL','WGTP')


# In[56]:

df_hhl_stot = build_table_cat_stot(pumas_hh_merge,'HHL','WGTP')


# ### Number of persons in HH

# In[57]:

df_np_sns = build_table_cat_sns(pumas_hh_merge,'NP','WGTP')


# In[58]:

df_np_stot = build_table_cat_stot(pumas_hh_merge,'NP','WGTP')


# In[59]:

df_np_stot.iloc[2]=['3-4',df_np_stot.iloc[2,1]+df_np_stot.iloc[3,1],df_np_stot.iloc[2,2]+df_np_stot.iloc[3,2]]


# In[60]:

df_np_stot.fillna(0,inplace=True)


# In[61]:

df_np_stot.iloc[3]=['5-8',df_np_stot.iloc[4,1]+df_np_stot.iloc[5,1]+df_np_stot.iloc[6,1]+df_np_stot.iloc[7,1],df_np_stot.iloc[4,2]+df_np_stot.iloc[5,2]+df_np_stot.iloc[6,2]+df_np_stot.iloc[7,2]]


# In[62]:

df_np_stot.iloc[4]=['9-14',df_np_stot.iloc[8,1]+df_np_stot.iloc[9,1]+df_np_stot.iloc[10,1]+df_np_stot.iloc[11,1]+df_np_stot.iloc[12,1]+df_np_stot.iloc[13,1],df_np_stot.iloc[8,2]+df_np_stot.iloc[9,2]+df_np_stot.iloc[10,2]+df_np_stot.iloc[11,2]+df_np_stot.iloc[12,2]+df_np_stot.iloc[13,2]]


# In[63]:

df_np_stot.drop([5,6,7,8,9,10,11,12,13,14],inplace=True)


# In[64]:

df_np_stot


# ### Household Type

# In[65]:

df_hht_sns = build_table_cat_sns(pumas_hh_merge,'HHT','WGTP')


# In[66]:

df_hht_stot = build_table_cat_stot(pumas_hh_merge,'HHT','WGTP')


# ### Family Type and Employment Status

# In[67]:

df_fes_sns = build_table_cat_sns(pumas_hh_merge,'FES','WGTP')


# In[68]:

df_fes_stot = build_table_cat_stot(pumas_hh_merge,'FES','WGTP')


# In[69]:

df_fes_stot.iloc[0]=['Married, 1+ employed',df_fes_stot.iloc[0,1]+df_fes_stot.iloc[1,1]+df_fes_stot.iloc[2,1],df_fes_stot.iloc[0,2]+df_fes_stot.iloc[1,2]+df_fes_stot.iloc[2,2]]
df_fes_stot.iloc[3,0]='Married, both unemployed'
df_fes_stot.iloc[4]=['Single, employed',df_fes_stot.iloc[4,1]+df_fes_stot.iloc[6,1],df_fes_stot.iloc[4,2]+df_fes_stot.iloc[6,2]]
df_fes_stot.iloc[5]=['Single, unemployed',df_fes_stot.iloc[5,1]+df_fes_stot.iloc[7,1],df_fes_stot.iloc[5,2]+df_fes_stot.iloc[7,2]]
df_fes_stot.drop([1,2,6,7,8],inplace=True)
df_fes_stot


# ### Presence and Age of Children

# In[70]:

df_hupac_sns = build_table_cat_sns(pumas_hh_merge,'HUPAC','WGTP')


# In[71]:

df_hupac_stot = build_table_cat_stot(pumas_hh_merge,'HUPAC','WGTP')


# In[72]:

df_hupac_stot.iloc[3,0] = 'No own children present'


# In[73]:

df_hupac_stot.iloc[4] = ['Own children present',df_hupac_stot.iloc[0,1]+df_hupac_stot.iloc[1,1]+df_hupac_stot.iloc[2,1],df_hupac_stot.iloc[0,2]+df_hupac_stot.iloc[1,2]+df_hupac_stot.iloc[2,2]]


# In[74]:

df_hupac_stot.drop([0,1,2],inplace=True)


# In[75]:

df_hupac_stot.reset_index(drop=True,inplace=True)


# In[76]:

df_hupac_stot


# ### Household Head's Ability to Speak English

# In[77]:

df_eng_sns = build_table_cat_sns(pumas_hh_merge,'ENG','WGTP')


# In[78]:

df_eng_stot = build_table_cat_stot(pumas_hh_merge,'ENG','WGTP')


# ### Household Head's Military Service Background

# In[79]:

df_mil_sns = build_table_cat_sns(pumas_hh_merge,'MIL','WGTP')


# In[80]:

df_mil_stot = build_table_cat_stot(pumas_hh_merge,'MIL','WGTP')


# ### Household Head's Sex

# In[84]:

df_sex_sns = build_table_cat_sns(pumas_hh_merge,'SEX','WGTP')


# In[85]:

df_sex_stot = build_table_cat_stot(pumas_hh_merge,'SEX','WGTP')


# In[86]:

df_sex_stot.drop(2,inplace=True)


# In[87]:

ind = np.arange(len(df_sex_stot))
width = 0.35 
fig, ax = plt.subplots()
rects1 = ax.bar(ind, df_sex_stot['% of total population'], width, color='b',alpha=0.3)
rects2 = ax.bar(ind + width, df_sex_stot['% of SNAP recipients'], width, color='r',alpha=0.3)
ax.set_ylabel('Percent (%) of population',fontsize=14)
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Men', 'Women'),fontsize=14)
ax.legend((rects1[0], rects2[0]), ('Total Population', 'SNAP Population'),fontsize=14)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+1.05,
                '%s' %str(height),
                ha='center', va='bottom',fontsize=12,fontweight='bold')
autolabel(rects1)
autolabel(rects2)
plt.grid('on')
plt.savefig('outputs/Bar_Household_Head_Sex.png')
plt.show()


# ### Household Head's Health Insurance Coverage

# In[88]:

df_hicov_sns = build_table_cat_sns(pumas_hh_merge,'HICOV','WGTP')


# In[89]:

df_hicov_stot = build_table_cat_stot(pumas_hh_merge,'HICOV','WGTP')


# ### Household Head's Nativity

# In[90]:

df_nat_sns = build_table_cat_sns(pumas_hh_merge,'NATIVITY','WGTP')


# In[91]:

df_nat_stot = build_table_cat_stot(pumas_hh_merge,'NATIVITY','WGTP')


# In[92]:

df_nat_stot.drop(2,inplace=True)
ind = np.arange(len(df_nat_stot))
width = 0.35 
fig, ax = plt.subplots()
rects1 = ax.bar(ind, df_nat_stot['% of total population'], width, color='b',alpha=0.3)
rects2 = ax.bar(ind + width, df_nat_stot['% of SNAP recipients'], width, color='r',alpha=0.3)
ax.set_ylabel('Percent (%) of population',fontsize=14)
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Native', 'Foreign born'),fontsize=14)
ax.legend((rects1[0], rects2[0]), ('Total Population', 'SNAP Population'),fontsize=14)
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height+1.05,
                '%s' %str(height),
                ha='center', va='bottom',fontsize=12,fontweight='bold')
autolabel(rects1)
autolabel(rects2)
plt.grid('on')
plt.savefig('outputs/Bar_Household_Head_Nativity.png')
plt.show()


# ### Household Head's Race

# In[93]:

df_race_sns = build_table_cat_sns(pumas_hh_merge,'RAC1P','WGTP')


# In[94]:

df_race_stot = build_table_cat_stot(pumas_hh_merge,'RAC1P','WGTP')


# ### Household Head's Means of Transportation to Work

# In[95]:

df_jwtr_sns = build_table_cat_sns(pumas_hh_merge,'JWTR','WGTP') 


# In[96]:

df_jwtr_stot = build_table_cat_stot(pumas_hh_merge,'JWTR','WGTP') 


# In[97]:

df_jwtr_stot.iloc[0] = ['Car, Truck, Van, or Motorcycle',df_jwtr_stot.iloc[0,1]+df_jwtr_stot.iloc[7,1],df_jwtr_stot.iloc[0,2]+df_jwtr_stot.iloc[7,2]]


# In[98]:

df_jwtr_stot.iloc[1] = ['Bus, Trolley/Streetcar, Subway, or Ferry',df_jwtr_stot.iloc[1,1]+df_jwtr_stot.iloc[2,1]+df_jwtr_stot.iloc[3,1]+df_jwtr_stot.iloc[4,1]+df_jwtr_stot.iloc[5,1],df_jwtr_stot.iloc[1,2]+df_jwtr_stot.iloc[2,2]+df_jwtr_stot.iloc[3,2]+df_jwtr_stot.iloc[4,2]+df_jwtr_stot.iloc[5,2]]


# In[99]:

df_jwtr_stot.iloc[6,0] = 'Taxicab'


# In[100]:

df_jwtr_stot.iloc[8,0] = 'Bicycle'


# In[101]:

df_jwtr_stot.iloc[9,0] = 'Walked'


# In[102]:

df_jwtr_stot.iloc[10,0] = 'Worked at home'


# In[103]:

df_jwtr_stot.iloc[11,0] = 'Other'


# In[104]:

df_jwtr_stot = df_jwtr_stot.drop([2,3,4,5,7,12])


# In[105]:

df_jwtr_stot


# ### Household Language Detailed

# In[106]:

df_hhlanp_stot = build_table_cat_stot(pumas_hh_merge,'HHLANP','WGTP')


# In[107]:

df_hhlanp_stot = df_hhlanp_stot[:-1]
df_hhlanp_stot.sort_values(ascending=False,by='% of SNAP recipients',axis=0,inplace=True)
print(df_hhlanp_stot.head(10))
df_hhlanp_stot.sort_values(ascending=False,by='% of total population',axis=0,inplace=True)
print(df_hhlanp_stot.head(10))


# SNAP recipients top 10 HHLANPs: 
# 1. 9500 = English-only household
# 2. 1200 = Spanish
# 3. 1250 = Russian
# 4. 1960 = Vietnamese
# 5. 4840 = Somali
# 6. 2920 = Tagalog
# 7. 1970 = Chinese
# 8. 2575 = Korean
# 9. 1900 = Khmer
# 10. 1170 = French
# 
# Total Population Top 10 HHLANPs: 
# 1. 9500 = English-only household
# 2. 1200 = Spanish
# 3. 1970 = Chinese
# 4. 1960 = Vietnamese
# 5. 2920 = Tagalog
# 6. 1250 = Russian
# 7. 2575 = Korean
# 8. 1110 = German
# 9. 1170 = French
# 10. 1350 = Hindi

# ### Household Head's Occupation

# In[108]:

df_occp_sns = build_table_cat_sns(pumas_hh_merge,'OCCP','WGTP')


# In[109]:

df_occp_stot = build_table_cat_stot(pumas_hh_merge,'OCCP','WGTP')
df_occp_stot = df_occp_stot[:-1]
df_occp_stot.sort_values(ascending=False,by='% of SNAP recipients',axis=0,inplace=True)
print(df_occp_stot.head(10))
df_occp_stot.sort_values(ascending=False,by='% of total population',axis=0,inplace=True)
print(df_occp_stot.head(10))


# SNAP recipients top 10 OCCPs: 
# 1. 4720 = Cashiers
# 2. 6050 = Miscellaneous Agricultural Workers, Including Animal Breeders
# 3. 4610 = Personal Care Aides
# 4. 9130 = Driver/Sales Workers And Truck Drivers
# 5. 5240 = Customer Service Representatives
# 6. 4020 = Cooks
# 7. 4220 = Janitors And Building Cleaners
# 8. 4760 = Retail Salespersons
# 9. 4230 = Maids And Housekeeping Cleaners
# 10. 430 = Miscellaneous Managers, Including Funeral Service Managers And Postmasters And Mail Superintendents
# 
# Total Population Top 10 OCCPs: 
# 1. 430 = Miscellaneous Managers, Including Funeral Service Managers And Postmasters And Mail Superintendents
# 2. 1020 = Software Developers, Applications And Systems Software
# 3. 2310 = Elementary And Middle School Teachers
# 4. 9130 = Driver/Sales Workers And Truck Drivers
# 5. 3255 = Registered Nurses
# 6. 4700 = First-Line Supervisors Of Retail Sales Workers
# 7. 5700 = Secretaries And Administrative Assistants
# 8. 5240 = Customer Service Representatives
# 9. 4220 = Janitors And Building Cleaners
# 10. 4760 = Retail Salespersons
