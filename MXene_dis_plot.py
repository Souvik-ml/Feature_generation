#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


df0 = pd.read_excel(r'C:\Users\DELL\Desktop\main_pred.xlsx')
df0


# In[46]:


df1 = df0.copy()


# In[48]:


S8_df1 = df1[df1['Al2Sn'] == 'S8'].copy()
S8_df1


# In[50]:


Al2S3_df1 = df1[df1['Al2Sn'] == 'Al2S3'].copy()
Al2S6_df1 = df1[df1['Al2Sn'] == 'Al2S6'].copy()
Al2S12_df1 = df1[df1['Al2Sn'] == 'Al2S12'].copy()
Al2S18_df1 = df1[df1['Al2Sn'] == 'Al2S18'].copy()
print(Al2S3_df1.shape, Al2S6_df1.shape, Al2S12_df1.shape, Al2S18_df1.shape)


# In[56]:


T = list(Al2S3_df1['T1'])
unique_T = list(Al2S3_df1['T1'].unique())
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [6, 4]
binding_energy = list(Al2S3_df1['ML Predicted BE (eV)'])
#T = list(df1['T1'])

colors_map = {'F':'#1f77b4', 'Cl':'#ff7f0e', 'Br':'#2ca02c', 'O':'#d62728', 'SCN':'#9467bd', 'NCS':'#8c564b', 'NCO':'#e377c2', 'PO':'#7f7f7f', 'OCN':'#bcbd22'}

#unique_T = list(set(T))
colors = [colors_map[t] for t in unique_T]

# Create the scatter plot
for i, t in enumerate(unique_T):
    indices = [idx for idx, val in enumerate(T) if val == t]
    plt.scatter(
        [T[idx] for idx in indices],
        [binding_energy[idx] for idx in indices],
        c='white',
        label=t,
        facecolors='white',       # Remove fill in the center
        edgecolors=colors[i],    # Set the edge color
        linewidths=1.5             # Set the edge thickness
    )
    
plt.xlabel('Terminal Groups', fontsize=20)
plt.ylabel('BE (eV)', fontsize=20)
#plt.title('Density Plot of ML Predicted BE for C and N', fontsize=18)

plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
# Show the legend
#plt.legend(loc="upper left", ncol=2, fontsize=15)
plt.savefig(r'C:\Users\DELL\Desktop\2.7-4.1\plot\bar plot\Al2S3_new_dis_T.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[ ]:





# In[ ]:





# In[47]:


import seaborn as sns


# In[13]:


import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
sns.kdeplot(data=df1[df1['Al2Sn'] == 'S8']['ML Predicted BE (eV)'], color='#1f77b4', label='S8', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S3']['ML Predicted BE (eV)'], color='#ff7f0e', label='Al2S3', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S6']['ML Predicted BE (eV)'], color='#2ca02c', label='Al2S6', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S12']['ML Predicted BE (eV)'], color='#d62728', label='Al2S12', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S18']['ML Predicted BE (eV)'], color='#9467bd', label='Al2S18', linewidth=2, fill=True)
plt.axvspan(-2.7, -4.1, color='lightgray', alpha=0.5, label='OBE')
plt.legend(['$S_8$', '$Al_2S_3$', '$Al_2S_6$', '$Al_2S_{12}$', '$Al_2S_{18}$'], loc="upper left", ncol=1, fontsize=15)
# Set plot labels and title
#plt.xlabel('ML Predicted BE (eV)', fontsize=30)
#plt.ylabel('Density', fontsize=30)

#x = np.linspace(1, 10, 100)
#y = np.sin(x)
#plt.plot(x, y)
plt.xlabel('ML Predicted BE (eV)', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.xticks(range(-8, 1), fontsize=15)

plt.yticks(fontsize=15)
plt.savefig('density_Al2Sn.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[15]:


import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True

sns.kdeplot(data=df1[df1['X'] == 'C']['ML Predicted BE (eV)'], color='#008080', label='C', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['X'] == 'N']['ML Predicted BE (eV)'], color='#FF6F61', label='N', linewidth=2, fill=True)

# Set plot labels and title
plt.xlabel('ML Predicted BE (eV)', fontsize=20)
plt.ylabel('Density', fontsize=20)
#plt.title('Density Plot of ML Predicted BE for C and N', fontsize=18)

plt.xticks(range(-8, 1), fontsize=15)
plt.yticks(fontsize=15)
# Show the legend
plt.legend(loc="upper left", fontsize=15)
plt.savefig('density_X.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[16]:


import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True

sns.kdeplot(data=df1[df1['T1'] == 'Br']['ML Predicted BE (eV)'], color='#1f77b4', label='Br', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'Cl']['ML Predicted BE (eV)'], color='#ff7f0e', label='Cl', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'F']['ML Predicted BE (eV)'], color='#2ca02c', label='F', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'NCO']['ML Predicted BE (eV)'], color='#d62728', label='NCO', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'NCS']['ML Predicted BE (eV)'], color='#9467bd', label='NCS', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'O']['ML Predicted BE (eV)'], color='#8c564b', label='O', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'OCN']['ML Predicted BE (eV)'], color='#e377c2', label='OCN', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'PO']['ML Predicted BE (eV)'], color='#7f7f7f', label='PO', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['T1'] == 'SCN']['ML Predicted BE (eV)'], color='#bcbd22', label='SCN', linewidth=2, fill=True)

# Set plot labels and title
plt.xlabel('ML Predicted BE (eV)', fontsize=20)
plt.ylabel('Density', fontsize=20)
#plt.title('Density Plot of ML Predicted BE for C and N', fontsize=18)

plt.xticks(range(-8, 1), fontsize=15)
plt.yticks(fontsize=15)
# Show the legend
plt.legend(loc="upper left", ncol=2, fontsize=15)
plt.savefig('density_T.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[18]:


T = list(df1['T1'])


# In[19]:


unique_T = list(df1['T1'].unique())


# In[29]:


import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [6, 4]
binding_energy = list(df1['ML Predicted BE (eV)'])
#T = list(df1['T1'])

colors_map = {'F':'#1f77b4', 'Cl':'#ff7f0e', 'Br':'#2ca02c', 'O':'#d62728', 'SCN':'#9467bd', 'NCS':'#8c564b', 'NCO':'#e377c2', 'PO':'#7f7f7f', 'OCN':'#bcbd22'}

#unique_T = list(set(T))
colors = [colors_map[t] for t in unique_T]

# Create the scatter plot
for i, t in enumerate(unique_T):
    indices = [idx for idx, val in enumerate(T) if val == t]
    plt.scatter(
        [T[idx] for idx in indices],
        [binding_energy[idx] for idx in indices],
        c='white',
        label=t,
        facecolors='white',       # Remove fill in the center
        edgecolors=colors[i],    # Set the edge color
        linewidths=1.5             # Set the edge thickness
    )
    
plt.xlabel('Terminal Groups', fontsize=20)
plt.ylabel('BE (eV)', fontsize=20)
#plt.title('Density Plot of ML Predicted BE for C and N', fontsize=18)

plt.xticks(fontsize=15, rotation=90)
plt.yticks(fontsize=15)
# Show the legend
#plt.legend(loc="upper left", ncol=2, fontsize=15)
plt.savefig('dis_T_new.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[26]:


S = list(df1['Al2Sn'])
unique_S = list(df1['Al2Sn'].unique())


# In[42]:


binding_energy = list(df1['ML Predicted BE (eV)'])
#T = list(df1['T1'])

colors_map = {'S8':'#1f77b4', 'Al2S3':'#ff7f0e', 'Al2S6':'#2ca02c', 'Al2S12':'#d62728', 'Al2S18':'#9467bd'}
replacement_map = {'S8': 'S₈', 'Al2S3': 'Al₂S₃', 'Al2S6': 'Al₂S₆', 'Al2S12': 'Al₂S₁₂', 'Al2S18': 'Al₂S₁₈'}
#unique_T = list(set(T))
colors = [colors_map[s] for s in unique_S]
#fig, ax = plt.subplots(
# Create the scatter plot
for i, s in enumerate(unique_S):
    indices = [idx for idx, val in enumerate(S) if val == s]
    plt.scatter(
        [S[idx] for idx in indices],
        [binding_energy[idx] for idx in indices],
        c='white',
        label=s,
        facecolors='white',       # Remove fill in the center
        edgecolors=colors[i],    # Set the edge color
        linewidths=1.5             # Set the edge thickness
    )


# Add labels and title
#l = ['S$_8$', 'Al$_2$S$_3$', 'Al$_2$S$_6$', 'Al$_2$S$_{12}$', 'Al$_2$S$_{18}$']
plt.xlabel(r'Al$_2$S$_n$', fontsize = 20)
plt.ylabel('BE (eV)', fontsize = 20)
#plt.title('Binding Energy across Terminal Groups')
plt.xticks(fontsize=15)
#plt.set_xlabels(['Al$_2$S$_8$', 'Al$_2$S$_3$', 'Al$_2$S$_6$', 'Al$_2$S$_{12}$', 'Al$_2$S$_{18}$'])
plt.yticks(fontsize=15)

subscript_labels = [replacement_map.get(s, s) for s in unique_S]
plt.xticks(range(len(unique_S)), subscript_labels)  # Set formatted tick labels
# Show the legend
#plt.legend(loc="upper left", ncol=2, fontsize=15)
plt.savefig('dis_Al2Sn2.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()
plt.show()


# In[43]:


X = list(df1['X'])
unique_X = list(df1['X'].unique())
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [6, 4]
binding_energy = list(df1['ML Predicted BE (eV)'])
#T = list(df1['T1'])

binding_energy = list(df1['ML Predicted BE (eV)'])
#T = list(df1['T1'])

colors_map = {'C':'#008080', 'N':'#FF6F61',}

#unique_T = list(set(T))
colors = [colors_map[x] for x in unique_X]

# Create the scatter plot
for i, x in enumerate(unique_X):
    indices = [idx for idx, val in enumerate(X) if val == x]
    plt.scatter(
        [X[idx] for idx in indices],
        [binding_energy[idx] for idx in indices],
        c='white',
        label=x,
        facecolors='white',       # Remove fill in the center
        edgecolors=colors[i],    # Set the edge color
        linewidths=1.5             # Set the edge thickness
    )


# Add labels and title
plt.xlabel('X', fontsize = 20)
plt.ylabel('BE (eV)', fontsize = 20)
#plt.title('Binding Energy across Terminal Groups')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# Show the plot
plt.show()
plt.savefig('X_S.png', dpi=700, bbox_inches="tight")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import tensorflow as tf 
from tensorflow import keras


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pd.set_option('display.max_columns', None)
df0 = pd.read_excel(r'C:\Users\DELL\Desktop\main_pred.xlsx')
df0


# In[5]:


df1 = df0.copy()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df1 is your DataFrame containing the data
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.size'] = 11.5
plt.rcParams["font.weight"]="bold"
plt.rcParams["axes.labelweight"]="bold"
# Set the style for the plot (white background without grid)
sns.set_style("whitegrid")

# Create the density plot with distinct colors
#plt.figure(figsize=(6, 4.5))
sns.kdeplot(data=df1[df1['Al2Sn'] == 'S8']['ML Predicted BE (eV)'], color='#1f77b4', label='S8', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S3']['ML Predicted BE (eV)'], color='#ff7f0e', label='Al2S3', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S6']['ML Predicted BE (eV)'], color='#2ca02c', label='Al2S6', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S12']['ML Predicted BE (eV)'], color='#d62728', label='Al2S12', linewidth=2, fill=True)
sns.kdeplot(data=df1[df1['Al2Sn'] == 'Al2S18']['ML Predicted BE (eV)'], color='#9467bd', label='Al2S18', linewidth=2, fill=True)
plt.axvspan(-2.7, -4.1, color='lightgray', alpha=0.5, label='OBE')
plt.legend(['$S_8$', '$Al_2S_3$', '$Al_2S_6$', '$Al_2S_{12}$', '$Al_2S_{18}$'], loc="upper left", ncol=1, fontsize=15)
# Set plot labels and title
plt.xlabel('ML Predicted BE (eV)', fontsize=30)
plt.ylabel('Density', fontsize=30)


plt.xticks(fontsize=20)
plt.yticks(fontsize=20)


#plt.savefig('density_Al2Sn.png', dpi=700, bbox_inches="tight", transparent=True)
# Show the plot
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
data1 = np.random.normal(0, 1, 500)
data2 = np.random.normal(2, 1, 500)
data3 = np.random.normal(-2, 0.5, 500)

# Combine the data
all_data = np.concatenate([data1, data2, data3])
labels = ['Data 1', 'Data 2', 'Data 3']

# Set style and create the density plot
sns.set(style="white")
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data1, color='#1f77b4', label='Data 1', linewidth=2, fill=True)
sns.kdeplot(data=data2, color='#ff7f0e', label='Data 2', linewidth=2, fill=True)
sns.kdeplot(data=data3, color='#2ca02c', label='Data 3', linewidth=2, fill=True)

# Set plot labels and title
plt.xlabel('X', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Density Plot using KDE', fontsize=16)

# Add legend
plt.legend(labels, loc='upper left')

# Set tick parameters
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save the plot
plt.savefig('density_plot.png', dpi=300, bbox_inches="tight", transparent=True)

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


Al = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Al\Al.xlsx')
Al


# In[5]:


x =Al['Voltage (V)']
x


# In[6]:


v = []
a = 0
while a < 264:
    v.append(sum(x[a:a+4])/4)
    a = a+4
print(v)


# In[7]:


len(v)


# In[8]:


avg_voltage = pd.DataFrame()
avg_voltage['Al_avg_v'] = v
avg_voltage


# In[9]:


y = Al['Solvent_name']
y


# In[10]:


avg_voltage['Solvent'] = y.unique()


# In[11]:


avg_voltage


# In[12]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Al\Al_voltage.xlsx')


# In[13]:


df_ca = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Ca\Ca.xlsx')
df_ca


# In[14]:


x = df_ca['Voltage (V)']
x


# In[15]:


v = []
a = 0
while a < 264:
    v.append(sum(x[a:a+4])/4)
    a = a+4
print(v)


# In[16]:


len(v)


# In[17]:


avg_voltage = pd.DataFrame()
avg_voltage['Ca_avg_v'] = v
avg_voltage['Solvent'] = y.unique()
avg_voltage


# In[18]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Ca\Ca_voltage.xlsx')


# In[19]:


#Cs = pd.read_excel(r'F:\project\p2\DROPBOX-26.9.2022\p2\New folder\Average voltage\Cs\vs-Cs.xlsx') 
K = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\K\K.xlsx')
Li = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Li\Li.xlsx') 
Mg = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Mg\Mg.xlsx')
Na = pd.read_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Na\Na.xlsx')
#Rb = pd.read_excel(r'F:\project\p2\DROPBOX-26.9.2022\p2\New folder\Average voltage\Rb\vs-Rb.xlsx')
#Y = pd.read_excel(r'F:\project\p2\DROPBOX-26.9.2022\p2\New folder\Average voltage\Y\vs-Y.xlsx')
#Zn = pd.read_excel(r'F:\project\p2\DROPBOX-26.9.2022\p2\New folder\Average voltage\Zn\vs-Zn.xlsx')


# In[79]:


Cs.head()


# In[20]:


K.head()


# In[21]:


Li.head()


# In[22]:


Mg.head()


# In[83]:


Zn.head()


# In[23]:


#x_Cs = Cs['Voltage (V)'] 
x_K = K['Voltage (V)']
x_Li = Li['Voltage (V)']
x_Mg = Mg['Voltage (V)']
x_Na = Na['Voltage (V)']
#x_Rb = Rb['Voltage (V)']
#x_Y = Y['Voltage (V)']
#x_zn = Zn['Voltage (V)']
S = y.unique()
S


# In[24]:


x_K


# In[25]:


v = []
a = 0
while a < 264:
    v.append(sum(x_K[a:a+4])/4)
    a = a+4
print(v)


# In[26]:


len(v)


# In[27]:


avg_voltage = pd.DataFrame()
avg_voltage['K_avg_v'] = v
avg_voltage['Solvent'] = S
avg_voltage


# In[28]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\K\K_voltage.xlsx')


# In[29]:


x_Li


# In[31]:


v = []
a = 0
while a < 264:
    v.append(sum(x_Li[a:a+4])/4)
    a = a+4
print(v)


# In[32]:


avg_voltage = pd.DataFrame()
avg_voltage['Li_avg_v'] = v
avg_voltage['Solvent'] = S
avg_voltage


# In[33]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Li\Li_voltage.xlsx')


# In[34]:


x_Mg


# In[35]:


v = []
a = 0
while a < 264:
    v.append(sum(x_Mg[a:a+4])/4)
    a = a+4
print(v)


# In[36]:


avg_voltage = pd.DataFrame()
avg_voltage['Mg_avg_v'] = v
avg_voltage['Solvent'] = S
avg_voltage


# In[37]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Mg\Mg_voltage.xlsx')


# In[38]:


x_Na


# In[39]:


v = []
a = 0
while a < 264:
    v.append(sum(x_Na[a:a+4])/4)
    a = a+4
print(v)


# In[40]:


avg_voltage = pd.DataFrame()
avg_voltage['Na_avg_v'] = v
avg_voltage['Solvent'] = S
avg_voltage


# In[41]:


avg_voltage.to_excel(r'G:\My Drive\p2_final\Plot\Average voltage plot\Na\Na_voltage.xlsx')


# In[88]:


Solvent = S.unique()
Solvent


# In[96]:


v = []
a = 0
while a < 264:
    v.append(sum(x_Cs[a:a+4])/4)
    a = a+4
print(v)
print(len(v))
avg_voltage = pd.DataFrame()
avg_voltage['Cs_avg_v'] = v
avg_voltage['Solvent'] = Solvent
print(avg_voltage)
avg_voltage.to_excel(r'F:\project\p2\DROPBOX-26.9.2022\p2\New folder\Average voltage\Cs\avg_voltage_Cs.xlsx')


# In[6]:


import numpy as np
ratios = np.arange(0.1, 0.51, 0.05)
ratios


# In[ ]:




