import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.io as pio
import plotly.figure_factory as ff

data = pd.read_csv('C:/Users/HP/OneDrive - metu.edu.tr/Masaüstü/MyPythonProjects/Data_Science/World_University_Ranking_Project/timesData.csv')
#print(data.head(3))


data.rename(columns={'university_name' : 'uni_name'},inplace=True)
df = data.iloc[:100,:]
print(data.columns)

pio.templates.default = 'simple_white'
'''
line1 = go.Scatter( x = df.world_rank,
                    y = df.citations,
                    mode='lines+markers',
                    name = 'citations',
                    marker = dict(color = 'rgba(78, 78, 250, 0.85)'),
                    text=df.uni_name)
line2 = go.Scatter( x = df.world_rank,
                    y = df.teaching,
                    mode='lines+markers',
                    name = 'teaching',
                    marker = dict(color = 'rgba(200, 45, 15, 0.85)'),
                    text=df.uni_name)          
mark1 = go.Scatter( x = df.world_rank,
                    y = df.citations,
                    mode='markers',
                    name = 'citations',
                    marker = dict(color = 'rgba(78, 78, 250, 0.85)'),
                    text=df.uni_name)
mark2 = go.Scatter( x = df.world_rank,
                    y = df.teaching,
                    mode='markers',
                    name = 'teaching',
                    marker = dict(color = 'rgba(200, 45, 15, 0.85)'),
                    text=df.uni_name)         

data1 = [mark1, mark2]
data = [line1, line2]
residential = dict(title = 'Citation and Education Scores of Top 100 Universities in the World Ranking', xaxis = dict(title = 'World Ranking', ticklen = 5, zeroline = False))
fig = dict(data=data1, layout=residential)
plot(fig, filename='1_line-citations and teaching scores.html')
'''

'''
data2011 = data[data.year == 2011].iloc[:5,:]
line1 = go.Bar( x = data2011.uni_name,
                    y = data2011.citations,
                    name = 'citations',
                    marker = dict(color = 'rgba(255, 127, 40, 0.5)',
                                line = dict(color='rgb(0,0,0)',width=1.5)),
                    text=data2011.country)
line2 = go.Bar( x = data2011.uni_name,
                    y = data2011.teaching,
                    name = 'teaching',
                    marker = dict(color = 'rgba(64, 12, 140, 0.5)',
                                line = dict(color='rgb(0,0,0)',width=1.5)),
                    text=data2011.country) 

data_ = [line1, line2]
residential = go.Layout(barmode = 'group')
fig = go.Figure(data=data_, layout=residential)
plot(fig, filename='2_line-citations and teaching scores.html')

'''

'''
data2011 = df[df.year==2011].iloc[:8,:]
slice1 = data2011.num_students
slice1_list = [float(each.replace(',','.')) for each in data2011.num_students]
labels = data2011.uni_name

line = go.Pie(labels = labels,
                values = slice1_list,
                hoverinfo='label+value+percent', #üzerinde gezinirken neleri görelim
                textinfo='value+percent',
                textfont=dict(size=15),
                rotation=180,
                hole=0.3,
                marker=dict(line=dict(color='#000000', width=1)))
data_ = [line]
residential = dict(title = 'year 2011 - student number and ratios of first 8 university',
                    legend = dict(orientation = 'h'))

fig = dict(data = data_, layout = residential)
plot(fig, filename = '3_line-num_students.html')
'''

data2011 = df[df.year==2011].iloc[:20,:]
num_students = [float(each.replace(',','.')) for each in data2011.num_students]

international_color = [float(each) for each in data2011.international]

data = [{'y': data2011.teaching,
        'x': data2011.world_rank,
        'mode' : 'markers',
        'marker': {'color': international_color,
                    'size': num_students,
                    'showscale': True},
        'text': data2011.uni_name}]
plot(data,filename='4_bubble-num_students.html')
