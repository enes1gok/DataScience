import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/penguin_project/penguins_size.csv')
#sns.heatmap(df.corr(numeric_only=True), annot = True, cmap = 'coolwarm')
#plt.show()

nan_percentage = df.isna().sum()/df.count()* 100
nan_count = df.isna().sum()

nan_table = pd.concat([nan_count,nan_percentage],axis=1)
nan_table.columns = ['Count','Percentage']
#print(nan_table)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")

df.iloc[:,:] = imputer.fit_transform(df)
#print(df.isna().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["gender"] = le.fit_transform(df["sex"])
#print(df["sex"].value_counts())
df = df.drop(labels=["sex"],axis=1)
species_count = df["species"].value_counts().reset_index()
#sns.barplot(data = species_count, x = 'index', y = 'species')
#plt.show()
df[df['species']=='Adelie']['body_mass_g']
#sns.kdeplot(df[df['species']=='Adelie']['body_mass_g'])
#sns.kdeplot(df[df['species']=='Gentoo']['body_mass_g'])
#sns.kdeplot(df[df['species']=='Chinstrap']['body_mass_g'])
'''
for col in df.columns[2:-1]:
    for spec in df['species'].unique():
        sns.kdeplot(df[df['species']==spec][col], shade = True, label=spec)
        plt.legend()
    plt.show()
'''
sns.pairplot(df, hue = 'species', height=3, diag_kind='hist')
plt.show()