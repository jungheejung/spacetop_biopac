
# %% load libraries 
import plotly.graph_objects as go # or plotly.express as px
from operator import index
import os, glob
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

final_df = pd.read_csv('/Users/h/Desktop/biopac_ttl_onset.csv')
final_df.reset_index(inplace=True)
final_df.rename( columns={'Unnamed: 0':'stacked_trials'}, inplace=True )

# %% PLOTLY
plotly_df = final_df[["ttl_r1", "ttl_r2", "ttl_r3", "ttl_r4", "stacked_trials"]]

melt_df = pd.melt(plotly_df, 
id_vars = 'stacked_trials', 
var_name = 'ttl', 
value_name = 'onset')
fig_plotly = px.scatter(data_frame = melt_df, 
                        x='onset', y='stacked_trials', 
                        color = 'ttl', range_x= [-3,18], 
                        marginal_x='box', 
                        size_max = 5
                        )
fig_plotly.update_xaxes(title_text='Pain trial TTL onset (s)')
fig_plotly.update_yaxes(title_text='trial No.')
lines = {'ttl1':2,'ttl2':7,'ttl3':9}
line_col = {'ttl1':'red','ttl2':'green','ttl3':'magenta'}
legend_col = {'ttl_r1':'1st TTL','ttl_r2':'2nd TTL','ttl_r3':'3rd TTL','ttl_r4':'4th TTL',}

for k in lines.keys():
    fig_plotly.add_shape(type='line',
                    yref="y",
                    xref="x",
                    x0=lines[k],
                    y0=-3,
                    x1=lines[k],
                    y1=melt_df['stacked_trials'].max()*1,
                    line={'dash': 'dash', 'color':line_col[k], 'width' :2}
                    )
fig_plotly.update_traces(marker=dict(size=3),
                  selector=dict(mode='markers'),
                  )

fig_plotly.for_each_trace(lambda t: t.update(name = legend_col[t.name],
                                      legendgroup = legend_col[t.name]
                                      
                                     )
                  )
fig_plotly.update_layout(
    width=800, height=1000,
    # margin=dict(l=20, r=20, t=20, b=20),
    # paper_bgcolor="LightSteelBlue",
)

# interactive dash _________________________________________

import io
from base64 import b64encode

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

buffer = io.StringIO()
app = dash.Dash(__name__)

fig_plotly.write_html(buffer)
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()
app.layout = html.Div([
    dcc.Graph(id="graph", figure=fig_plotly),
    html.A(
        html.Button("Download HTML"), 
        id="download",
        href="data:text/html;base64," + encoded,
        download="plotly_graph.html"
    )
])

app.run_server(debug=True)
