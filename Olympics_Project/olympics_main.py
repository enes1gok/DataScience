import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter

#To turn of warnings
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Olympics_Project/athlete_events.csv")
#print(data.head())

#print(data.info())

#print(data.columns)

data.rename(columns={'Sex':'Gender'},inplace=True) #inplace: It works on an existing column without creating a new column.
#print(data.head(2))

data = data.drop(['ID','Games'], axis=1) #I removed the unnecessary columns from the data. (axis = 1) means columns.
#print(data.head(2))

unique_events = pd.unique(data.Event) #The unique function gives how many different categories there are in categorical data.
#print("Number of unique events : {}".format(len(unique_events)))
#print(unique_events[:5])

backup_data = data.copy() #I copied the data just in case.

list_of_height_and_weight = ['Height','Weight']
for e in unique_events:
    data_filter = backup_data.Event == e            #backup_data.Event tüm sütünu dönmüyor mu? sütunu nasıl e variable ına eşitliyoruz????? 
    filtered_data = backup_data[data_filter]        #boşlukları o verinin olduğu eventinin diğer datalarındaki değerlerin ortalamasını alıp doldurduk.
    for s in list_of_height_and_weight:
        average = np.round(np.mean(filtered_data[s]),2)
        if ~np.isnan(average): # "~" means not. If average of Event is not nan.
            filtered_data[s] = filtered_data[s].fillna(average)
        else:
            average_of_all_data = np.round(np.mean(data[s]),2)
            filtered_data[s] = filtered_data[s].fillna(average_of_all_data)

    backup_data[data_filter] = filtered_data
data = backup_data.copy()
#print(data.head(5))
#print(data.info())

#If it has an average, I filled it with that, otherwise I took the average of the other data and filled it with it.

average_age = np.round(np.mean(data.Age),2)
#print("The average age is {}".format(average_age))
data['Age'] = data['Age'].fillna(average_age)

#print(data.Age)    #What is the difference of each other???
#print(data['Age']) #What is the difference of each other???

medal_variable = data['Medal']
#print(pd.isnull(medal_variable).sum())

medal_variable_filter = ~pd.isnull(medal_variable) #Select non-null values in the medal column and generate data only from those values.
data = data[medal_variable_filter]
#print(data.head(10))
#print(data.info())

data.to_csv("C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/Olympics_Project/cleaned_olympics.csv", index = False)

'''
Numerical Datas: Age - Height - Weight - Year
Histogram: To examine the frequency of numerical data.
Box Plot: For interpreting basic Statistics information.
'''
def PlotHistogram(variable):                            #Input: Name of variable/Column
    plt.figure()                                        #Output: Histogram of the relevant variable
    plt.hist(data[variable],bins = 85, color='red')     #bins means that dividing 85 the variable 
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("Data Distribution - {}".format(variable))
    plt.show()

numerical_variables = ["Age","Height","Weight","Year"]

'''
for i in numerical_variables:    
   PlotHistogram(i)

print(data.describe())

plt.boxplot(data.Age)
plt.title("Boxplot Graph For Age Variable")
plt.xlabel("Age")
plt.ylabel("Value")
plt.show()

'''

#We will use bar charts to examine categorical variables.
def plotBar(variable, n=5):
    data_ = data[variable]
    data_count = data_.value_counts()
    data_count = data_count[:n]
    plt.figure()
    plt.bar(data_count.index, data_count, color = "red")
    plt.xticks(data_count.index, data_count.index.values, rotation = 30)
    plt.ylabel('Frequency')
    plt.title("Data Frequency - {}".format(variable))
    plt.show()
    print("{}: \n {}".format(variable, data_count))

categorical_variables = ["Name","Gender","Team", "NOC", "Season","City", "Sport", "Event","Medal"]
'''
for i in categorical_variables:
    plotBar(i)
'''

#                              bivariate data analysis
'''
male = data[data.Gender == "M"]
female = data[data.Gender == "F"]

plt.figure()
plt.scatter(male.Height, male.Weight, alpha = 0.4, label = "Male", color = "blue")
plt.scatter(female.Height, female.Weight, alpha = 0.4, label = "Female", color = "pink")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("The relationship between height and weight")
plt.legend()
plt.show()

print(data.loc[:,["Age", "Height", "Weight"]].corr())#correlation table loc means location, iloc means index location

'''
temporary_data = data.copy()
temporary_data = pd.get_dummies(temporary_data, columns=["Medal"])#get dummies: A separate column was opened for each medal.
#print(temporary_data.head(2))

temporary_data.loc[:,["Age","Medal_Bronze", "Medal_Gold", "Medal_Silver"]]

temporary_data[["Team","Medal_Bronze", "Medal_Gold", "Medal_Silver"]].groupby(["Team"], as_index = False).sum().sort_values(by="Medal_Gold",ascending = False)

temporary_data[["City","Medal_Bronze", "Medal_Gold", "Medal_Silver"]].groupby(["City"], as_index = False).sum().sort_values(by="Medal_Gold",ascending = False)[:10]

temporary_data[["Gender","Medal_Bronze", "Medal_Gold", "Medal_Silver"]].groupby(["Gender"], as_index = False).sum().sort_values(by="Medal_Gold",ascending = False)

#PIVOT TABLE ~~ multivariate data analysis

data_pivot = data.pivot_table(index="Medal", columns="Gender", values=['Height', 'Weight', 'Age'],
aggfunc={'Height':np.mean, 'Weight':np.mean, 'Age':[min, max, np.std]})
#print(data_pivot.head())
'''
index = It divided it into rows according to medals.
columns = It separated the variables that I defined into values as female and male in the columns.
'''

#Anomaly Detection and Outlier Detection

def anomalyDetection(df, feature):
    outlier_indices = []

    for c in feature:
        #first quarter
        Q1 = np.percentile(df[c],25)
        #third quarter
        Q3 = np.percentile(df[c],75)
        #interquartile range
        IQR = Q3 - Q1
        #amount of additional step for outlier value
        outlier_step = 1.5*IQR
        #Let's find outlier value and its index
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        #Storage of indices
        outlier_indices.extend(outlier_list_col)
    #Let's find unique outlier values of indices
    outlier_indices = Counter(outlier_indices)
    #If an example has different values in v columns, I consider this as an inconsistency.
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)
    return multiple_outliers

data_anomaly = data.loc[anomalyDetection(data,['Age','Weight', 'Height'])]
#print(data_anomaly.Sport.value_counts())

'''
plt.figure()
plt.bar(data_anomaly.Sport.value_counts().index, data_anomaly.Sport.value_counts().values)
plt.xticks(rotation = 30) #Tilt the xlabels 30 degrees.
plt.title("Sports Branches with Anomalies")
plt.ylabel("Frequency")
plt.grid(True, alpha = 0.6)                             #to create a grid inside the plot, alpha gives transparency
plt.show()
'''

data_gym = data_anomaly[data_anomaly.Sport == "Gymnastics"]
#print(data_gym)

#print(data_gym.Event.value_counts())
data_basketball = data_anomaly[data_anomaly.Sport == "Basketball"]
#print(data_basketball)
#print(data_basketball.Event.value_counts())

#            Data Analysis in time series

data_time = data.copy()
unique_years = data_time.Year.unique() #datetime: very useful in temporal(time) analysis, to measure increasing and decreasing trends over time
#print(unique_years)
sorted_array = np.sort(data_time.Year.unique())
#print(sorted_array)
'''
plt.figure()
plt.scatter(range(len(sorted_array)),sorted_array)
plt.grid(True)
plt.ylabel("Years")
plt.title("The Olympics are held in even years.")
plt.show()
'''
#Let's convert the year values in the data into datetime data types.
date_time_object = pd.to_datetime(data_time["Year"], format = "%Y")
#print(type(date_time_object))
#print(date_time_object.head(3))

data_time["date_time"] = date_time_object
#print(data_time.head(3))
data_time = data_time.set_index("date_time")  #By adjusting the indexes according to time, we opened the way to analyze them, so                                             the horizontal axis became time.
data_time.drop(["Year"], axis = 1, inplace = True)
#print(data_time.head(5))

periodic_data = data_time.resample("2A").mean() #It takes the average values in 2-year periods. We split the data into slices with resample
#print(periodic_data.head())

#extracting lost data
periodic_data.dropna(axis=0,inplace=True)
#print(periodic_data.head())

'''
plt.figure()
periodic_data.plot()
plt.title("Average Age, Height and Weight change according to Years")
plt.xlabel("Year")
plt.grid(True)
plt.show()
'''

data_time = pd.get_dummies(data_time, columns=['Medal']) 
#get dummies: It creates new columns according to the values that it can take categorical data.
#print(data_time.head(3))

periodic_data = data_time.resample("2A").sum()
periodic_data = periodic_data[~(periodic_data == 0).any(axis=1)]
#print(periodic_data.tail())
'''
plt.figure()
periodic_data.loc[:,["Medal_Bronze","Medal_Gold","Medal_Silver"]].plot()
plt.title("Medal Amount according to Years")
plt.ylabel("Number")
plt.xlabel("Year")
plt.grid(True)
plt.show()
'''

summer = data_time[data_time.Season == "Summer"]
winter = data_time[data_time.Season == "Winter"]

periodic_data_winter = winter.resample("A").sum()
periodic_data_winter = periodic_data_winter[~(periodic_data_winter == 0).any(axis=1)]
#print(periodic_data_winter.head())
periodic_data_summer = summer.resample("A").sum()
periodic_data_summer = periodic_data_summer[~(periodic_data_summer == 0).any(axis=1)]
#print(periodic_data_summer.head())
'''
plt.figure()
periodic_data_summer.loc[:,["Medal_Bronze","Medal_Gold", "Medal_Silver"]].plot() # We used plot because it is a time series.
plt.title("Number of medals by year - Summer Season")
plt.ylabel("Number")
plt.xlabel("Year")
plt.grid(True)
plt.show()

plt.figure()
periodic_data_winter.loc[:,["Medal_Bronze","Medal_Gold", "Medal_Silver"]].plot() # We used plot because it is a time series.
plt.title("Number of medals by year - Winter Season")
plt.ylabel("Number")
plt.xlabel("Year")
plt.grid(True)
plt.show()
'''


'''
mu1 = 100
sigma1 = 15
x = mu1 + (sigma1 * np.random.randn(10000))

mu2 = 100
sigma2 = 15
y = mu2 + (sigma2 * np.random.randn(10000))

n, bins, patches = plt.hist(x, 100, density = 1, facecolor = 'b', alpha = 0.25)
n, bins, patches = plt.hist(y, 100, density = 1, facecolor = 'g', alpha = 0.25)

plt.xlabel("Datas")
plt.ylabel("Probabilities")
plt.title(' Histogram of the data: $\mu_1 = 100,\ \mu_2 = 110$')
plt.annotate("Top of the Gaussian Curve", xy = (100,0.03), xytext = (120,0.04), arrowprops = dict(facecolor = "black", shrink = 0.1))
plt.text(60, 0.03, r'$\mu_1 = 100,\ \sigma_1 = 15$')
plt.text(120, 0.03, r'$\mu_2 = 110,\ \sigma_2 = 10$')
plt.axis([40, 160, 0, 0.05])
plt.grid(True)
plt.show()


x = np.arange(0, 10, 0.01)
y = np.exp(x)


plt.figure()
plt.subplot(121)
plt.plot(x,y)
plt.yscale('linear')
plt.title("Linear Graph")
plt.grid(True)

plt.subplot(122)
plt.plot(x,y)
plt.yscale('log')
plt.title("Logarithmic Graph")
plt.grid(True)

plt.show()
'''

'''
def scatterPlotDraw():
    sns.scatterplot(x = 'Height', y = 'Weight', data = data)
    plt.title('Relation of Height and Weight')
    plt.show()
#default = white - set_style
#scatterPlotDraw()
'''

'''
sns.set_style('whitegrid')
sns.scatterplot( x = 'Height', y = 'Weight', hue = 'Medal', data = data)
plt.title('Height and Weight distribution by Medal')
plt.show()

sns.scatterplot( x = 'Height', y = 'Weight', hue = 'Medal', data = data, palette = 'Set1' )
plt.title('Height and Weight distribution by Medal')
plt.show()

sns.set_style('whitegrid')
sns.scatterplot( x = 'Height', y = 'Weight', hue = 'Gender', data = data)
plt.title('Height and Weight distribution by Gender')
plt.show()

sns.set_style('whitegrid')
sns.scatterplot( x = 'Height', y = 'Weight', hue = 'Gender', style = 'Medal', data = data)
plt.title('Height and Weight distribution by Gender and Medal')
plt.show()

sns.set_style('whitegrid')
sns.scatterplot( x = 'Height', y = 'Weight', hue = 'Gender', style = 'Medal', size = 'Age', sizes = (15,200), data = data)
plt.title('Height and Weight distribution by Gender, Medal and Age')
plt.show()
'''

'''
sns.set_style('whitegrid')
sns.regplot( x = 'Height', y = 'Weight', data = data, marker = '+', scatter_kws = {'alpha': 0.2})
plt.title('Height and Weight distribution')
plt.show()
'''

'''
sns.lineplot( x = 'Height', y = 'Weight', hue = 'Season', data = data)
plt.title("Relation Height and Weight by Medal")
plt.show()
'''
'''
sns.displot( data, x = 'Weight', hue = 'Gender')
plt.ylabel('Frequency')
plt.title('Weight Histogram Disaggregated by gender')
plt.show()
'''

'''
sns.displot( data, x = 'Weight', col = 'Gender', multiple = 'dodge')
plt.show()

sns.displot(data, x = 'Weight', y = 'Height', kind = 'kde')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Weight - Height Histogram')
plt.show()

sns.displot(data, x = 'Weight', y = 'Height', hue = 'Gender', kind = 'kde')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Weight - Height Histogram disaggregated by gender')
plt.show()
'''

'''
sns.kdeplot(data = data, x = 'Weight', y = 'Height', hue = 'Gender', fill = True)
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Weight - Height Histogram disaggregated by gender')
plt.show()
'''

'''
sns.barplot( x = 'Medal', y = 'Height', data = data, hue = 'Gender')
plt.title("Height Charts by Medal")
plt.show()
'''

'''
sns.catplot(x = 'Medal', y = 'Age', hue = 'Gender', col = 'Season', data = data, kind = 'bar', height = 4, aspect = 1)
plt.show()
'''
'''
sns.boxplot(x = 'Season', y = 'Height', data = data)
plt.show()

sns.boxplot(x = 'Season', y = 'Height', hue = 'Gender', data = data)
plt.show()


temporary_data = data.loc[:,['Age','Weight','Height']]
sns.boxplot(data=temporary_data, orient='h',palette='rocket') #h means horizontal
plt.show()
'''

'''
sns.catplot(x = 'Height', y = 'Season', col = 'Medal', data = data, kind = 'box', height = 4, aspect=1, palette='rocket', orient='h')
plt.show()
'''

'''
sns.heatmap(data.corr(), annot = True, linewidths=0.5, fmt='.1f') #.1f noktadan sonra bir basamak olsun demek
plt.show()               #annot kutuların içinde sayıları göster demekdir.
'''

'''
sns.violinplot(x = 'Season', y = 'Height', hue = 'Gender', data = data, split=True)
plt.show()

sns.catplot(x='Season',y='Height', hue='Gender', col = 'Medal', data = data, kind = 'violin', split = True, height=4,aspect=0.6)
plt.show()
'''

'''
sns.jointplot(data = data, x = 'Weight', y = 'Height', hue = 'Season', kind = 'kde')
plt.show()

g = sns.JointGrid(data=data, x = 'Weight', y = 'Height')
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)
plt.show()
'''

'''
sns.pairplot(data)
plt.show()
'''
'''
g = sns.PairGrid(data)  #We named the graph-related variables with g.
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot,fill =True)
g.map_diag(sns.histplot, kde=True)
plt.show()
'''

'''
sns.countplot(x='City', data = data)
plt.xticks(rotation = 90)
plt.show()
'''