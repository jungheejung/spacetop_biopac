
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
legend_col = {'ttl_r1':'TTL 1','ttl_r2':'TTL 2','ttl_r3':'TTL 3','ttl_r4':'TTL 4',}

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
import flask

css_directory = os.getcwd()
stylesheets = ['plotly_biopac.css']
static_css_route = '/static/'



buffer = io.StringIO()
# external_stylesheets = ['https://www.w3schools.com/w3css/tryw3css_templates_architect.htm']
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] 
# external_stylesheets = ['/Users/h/Dropbox/projects_dropbox/spacetop_biopac/scripts/s02_extract_acq/plotly.css']
app = dash.Dash(__name__, assets_url_path = '/Users/h/Dropbox/projects_dropbox/spacetop_biopac/scripts/s02_extract_acq/assets/plotly_biopac.css') #, external_stylesheets=external_stylesheets)

fig_plotly.write_html(buffer)
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()
# import dash_core_components as dcc
# dcc._css_dist[0]['relative_package_path'].append('plotly_biopac.css')
app.layout = html.Div([
    dcc.Markdown("""
    Author: Heejung Jung \n
    Date: Jan 31, 2022 """, style = {'textAlign': 'right', 'margin-bottom':0}),
    dcc.Markdown("""


    # Medoc TSA2 onset times are delayed compared to intended program 

    ### How did we measure onset times from Medoc? *via TTL onsets*
    Using Biopac, we recorded TTL signals from Medoc. 
    In total, there are 4 TTLs: 
    * 1) TT1 1 (blue)  : Onset of when the program was triggered
    * 2) TT1 2 (orange): When program reaches Desitination Temperature
    * 3) TT1 3 (green) : When program terminates Desitination Temperature, i.e. end of Plateau
    * 4) TT1 4 (purple): When program ramps down to Baseline
    
    We compared the measured TTL onsets against the expected TTL onsets. 

    ### How are the "expected TTL onsets" calculated?
    Based on the Medoc programs that we built, we calculate the time to peak, time after plateau/duration, and time to reach baseline. NOTE: details of the program are listed below.
    * Expected TTL 1 (when triggered): **0s**
    * Expected TTL 2 (time to peak): (50-32 celcius) / 13 (C/sec) = 1.38s ≅ **2s**
    * Expected TTL 3 (after 5s duration): TTL 2 + 5 s = **7s**
    * Expected TTL 4 (time to baseline):  7 + 2 = **9s**


    

    """),

    dcc.Graph(id="graph", figure=fig_plotly),
    
    dcc.Markdown("""

    ### What are the plot elements?
    - X axis: Time in seconds. 0 is when a program — e.g. Matlab — triggers Medoc's program via external control
    - Y axis: Trials. 
        - Entire trials from 90 subjects x 6 runs x 12 trials are plotted. 
        - subjects and runs are not sorted
    - Dashed line: Each line (blue/orange/green/purple) indicates the "expected TTL onset"


    # Conclusion
    * Using **Manual for Trial** trigger method in Medoc, the first TTLs are always delivered on intended onset. (There is only one exception, out of 2000 trials)
    * The delay happens in two phases 1) TTL1 to TTL2 , 2) TTL3 to TTL4. In other words, reaching the destination temperature and reaching baseline temperature takes longer than expected.
    
    ## Future recommendations
    * We recommend that researchers utilize TTL equipment and collect the onsets of the TTLs whenever using Medoc-triggered stimuli.
    * Different programs may have different delay profiles. Here, we observe a 2.17s delay between TTL1 and TTL2. 
    We do not know if this is Medoc's average delay profile, or whether we would see different delay patterns for 
    lower destination temperatures, slower return rates, etc.We recommend that researchers collect pilot data to 
    identify delay profiles and use the TTLs to inform realistic estimations on each trial duration and the entire 
    experimental design.

    ---

    #### Appendix: features of the Medoc programs that we used
    The intended program has the following features, using Medoc software's terminology.
    * Baseline: 32
    * Time before sequence 0
    * Trigger: Manual for Trial
    * Destination Temperature (C): 48, 49, 50
    * Destination Criterion: Temperature
    * Duration Time (sec): 5
    * Return Option: Baseline
    * Return Rate (C/sec): 13
    * Number of Trials: 1


    """),
    html.A(
        html.Button("Download HTML"), 
        id="download",
        href="data:text/html;base64," + encoded,
        download="biopac_TTL.html"
    )
])


# @app.server.route('{}<stylesheet>'.format(static_css_route))
# def serve_stylesheet(stylesheet):
#     if stylesheet not in stylesheets:
#         raise Exception(
#             '"{}" is excluded from the allowed static files'.format(
#                 stylesheet
#             )
#         )
#     return flask.send_from_directory(css_directory, stylesheet)


app.run_server(debug=True)
