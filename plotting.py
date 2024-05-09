import streamlit as st
import pandas as pd
import math
import numpy as np
# Plotly imports
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from pickett import Pickett_plot
from hingle import Hingle_plot

def plot(las_file, well_data):
    st.title('LAS File Visualisation')
    well_data['PHIT'] = (2.69-well_data['DENS'])/(2.69-1)
    curve_units = {}
    for count, curve in enumerate(las_file.curves):
        curve_units[curve.mnemonic] = curve.unit
        
    if not las_file:
        st.warning('No file has been uploaded')
    
    else:
        columns = list(well_data.columns)
        st.write('Expand one of the following to visualise your well data.')
        st.write("""Each plot can be interacted with. To change the scales of a plot/track, click on the left hand or right hand side of the scale and change the value as required.""")
        with st.beta_expander('Log Plot'):
            curves = st.multiselect('Select Curves To Plot', columns)
            log_bool = False
            if len(curves) < 1:
                st.warning('Please select at least 1 curve.')
            else:
                curve_index = 1
                fig = make_subplots(rows=1, cols=len(curves), subplot_titles=curves, shared_yaxes=True)
                i = 0
                for curve in curves:
                    col1, col2 = st.beta_columns([3, 1])
                    log_scale = col1.checkbox(f'Logarithmic Scale for {curve}')
                    reverse_x = col2.checkbox(f'Reverse X-axis for {curve}')

                    x_data = np.log10(well_data[curve]) if log_scale else well_data[curve]

                    fig.add_trace(go.Scatter(x=x_data, y=(well_data['DEPTH']), name=curve), row=1, col=curve_index)

                    fig.update_xaxes(title_text=f'{curve} ({curve_units[str(curve)]})', row=1, col=curve_index)
                    fig['layout']['xaxis' + str(i + 1)].update(showline=True, linewidth=1, linecolor='black', showspikes=True, side='top', title_standoff=0)
                    fig['layout']['yaxis' + str(i + 1)].update(showline=True, linewidth=1, linecolor='black', showspikes=True)
                    fig['layout']['annotations'][i]['yref'] = 'paper'
                    fig['layout']['annotations'][i]['y'] = 1.05
                    if reverse_x:
                        fig['layout']['xaxis' + str(i + 1)].update(autorange="reversed")  # Reverse x-axis
                    # Calculate minor tick step (dtick)
                    i += 1
                    curve_index += 1

                fig.update_layout(height=1000, showlegend=True, yaxis={'title':f'Depth ({curve_units["DEPTH"]})','autorange':'reversed'})
                fig.layout.template = 'seaborn'
                st.plotly_chart(fig, use_container_width=True)
                fig['layout'].update(
                    legend=dict(x=-0.2, y=1),
                    hovermode='closest',
                    dragmode='select'
                )
        
        with st.beta_expander('OverLay Plot'):

            props_to_relate = st.multiselect('Select Overlay Curves to Plot', columns)
            log_option1 = st.radio('Select Linear or Logarithmic Scale for x1 axis', ('Linear', 'Logarithmic'))
            log_bool1 = False
            if log_option1 == 'Linear':
                log_bool1 = False
            else:
                log_bool1 = True

            log_option2 = st.radio('Select Linear or Logarithmic Scale for x2 axis', ('Linear', 'Logarithmic'))
            log_bool2 = False
            if log_option2 == 'Linear':
                log_bool2 = False
            else:
                log_bool2 = True

            if len(props_to_relate) != 2:
                st.warning('Please select exactly 2 curves')
            else:
                plot_title = '[' + props_to_relate[0] + ', ' + props_to_relate[1] + '] vs Depth'
                depth = well_data["DEPTH"]
                x1 = well_data[props_to_relate[0]]
                x2 = well_data[props_to_relate[1]]
                if log_bool1:
                    x1 = np.log10(well_data[props_to_relate[0]])
                if log_bool2:
                    x2 = np.log10(well_data[props_to_relate[1]])
                x_unit = curve_units["DEPTH"]
                data = []
                trace1 = go.Scatter(
                    x=x1,
                    y=depth,
                    name=props_to_relate[0],
                )
                trace2 = go.Scatter(
                    x=x2,
                    y=depth,
                    name=props_to_relate[1],
                    xaxis='x2',
                )
                data = [trace1, trace2]
                layout = go.Layout(
                    title=plot_title,
                    xaxis=dict(
                        title=f"{props_to_relate[0]} ({curve_units.get(str(props_to_relate[0]), 'Unknown')})",
                        showspikes=True,
                        linecolor='black',  # Set border color to black
                        mirror=True,  # Show ticks on both sides
                        ticks='outside',  # Place ticks outside the axis
                        ticklen=10,  # Length of ticks
                        tickwidth=1,  # Width of ticks
                        tickcolor='black',  # Color of ticks
                        nticks=5,  # Number of subticks
                        showgrid=True
                    ),
                    xaxis2=dict(
                        title=f"{props_to_relate[1]} ({curve_units.get(str(props_to_relate[1]), 'Unknown')})",
                        overlaying='x',
                        side='top',
                        showspikes=True,
                        linecolor='black',  # Set border color to black
                        mirror=True,  # Show ticks on both sides
                        ticks='outside',  # Place ticks outside the axis
                        ticklen=10,  # Length of ticks
                        tickwidth=1,  # Width of ticks
                        tickcolor='black',  # Color of ticks
                        nticks=5,  # Number of subticks
                        showgrid=True
                    ),
                    yaxis=dict(
                        title=f"Depth ({x_unit})",
                        autorange='reversed',
                        showspikes=True,
                        linecolor='black',  # Set border color to black
                        mirror=True,  # Show ticks on both sides
                        ticks='outside',  # Place ticks outside the axis
                        ticklen=10,  # Length of ticks
                        tickwidth=1,  # Width of ticks
                        tickcolor='black',  # Color of ticks
                        nticks=5,  # Number of subticks
                        showgrid=True
                    ),
                    autosize=True,
                    width=600,
                    margin=dict(
                        l=105,
                        r=50,
                        b=65,
                        t=150
                    ),
                )

                fig = go.Figure(data=data, layout=layout)
                fig.update_layout(height=1000, showlegend=True, yaxis={'title':f'Depth ({curve_units["DEPTH"]})','autorange':'reversed'}
                )
                fig.layout.template='seaborn'
                st.plotly_chart(fig, use_container_width=False)
                
                fig['layout'].update(
                    legend=dict(x=-0.2,
                                y=1, ),
                    hovermode='closest',
                    dragmode='select'
                )

                # buttons = [dict(method='restyle',
                # label='linear',
                # visible=True,
                # args=[{'label': 'linear',
                #        'visible':[True, False],
                #       }
                #      ]),
                # dict(method='restyle',
                #         label='log',
                #         visible=True,
                #         args=[{'label': 'log',
                #             'visible':[False, True],
                #             }
                #             ])
                #         ]
                # um = [{'buttons':buttons,
                #     'direction': 'down'}
                #     ]

                # fig.update_layout(updatemenus=um)
        with st.beta_expander('Histograms'):
            col1_h, col2_h = st.beta_columns(2)
            col1_h.header('Options')

            hist_curve = col1_h.selectbox('Select a Curve', columns)
            log_option = col1_h.radio('Select Linear or Logarithmic Scale', ('Linear', 'Logarithmic'), key=999)
            hist_col = col1_h.color_picker('Select Histogram Colour')
            st.write('Color is'+hist_col)
            
            if log_option == 'Linear':
                log_bool = False
            elif log_option == 'Logarithmic':
                log_bool = True
        

            histogram = px.histogram(well_data, x=hist_curve, log_x=log_bool)
            histogram.update_traces(marker_color=hist_col)
            histogram.layout.template='seaborn'
            col2_h.plotly_chart(histogram, use_container_width=True)

        with st.beta_expander('Crossplot'):
            col1, col2 = st.beta_columns(2)
            col1.write('Options')

            xplot_x = col1.selectbox('X-Axis', columns)
            xplot_y = col1.selectbox('Y-Axis', columns)
            xplot_col = col1.selectbox('Colour By', columns)
            xplot_x_log = col1.radio('X Axis - Linear or Logarithmic', ('Linear', 'Logarithmic'))
            xplot_y_log = col1.radio('Y Axis - Linear or Logarithmic', ('Linear', 'Logarithmic'))

            if xplot_x_log == 'Linear':
                xplot_x_bool = False
            elif xplot_x_log == 'Logarithmic':
                xplot_x_bool = True
            
            if xplot_y_log == 'Linear':
                xplot_y_bool = False
            elif xplot_y_log == 'Logarithmic':
                xplot_y_bool = True

            col2.write('Crossplot')
           
            xplot = px.scatter(well_data, x=xplot_x, y=xplot_y, color=xplot_col, log_x=xplot_x_bool, log_y=xplot_y_bool)
            xplot.layout.template='seaborn'
            col2.plotly_chart(xplot, use_container_width=True)

        with st.beta_expander('Pickettplot'):
            phit_gt_01 = well_data[well_data['PHIT'] > 0.1]
    
            if not phit_gt_01.empty:
                pickett = Pickett_plot(phit_gt_01['RESD'], phit_gt_01['PHIT'])
                st.set_option('deprecation.showPyplotGlobalUse', False)
                Rwa = st.slider('Rwa', min_value=0.0001, max_value=1.0, value=0.033, step=0.0001)
                m = st.slider('m', min_value=1.1, max_value=3.1, value=2.0, step=0.05)
                a = st.slider('a', min_value=0.5, max_value=2.0, value=1.0, step=0.05)
                n = st.slider('n', min_value=1.1, max_value=3.1, value=2.0, step=0.05)
                pickett.plot_saturation(Rwa, m, a, n)
                k = phit_gt_01['RESD'].transform(lambda x : (1/x)**(1/m))

                hingle = Hingle_plot(phit_gt_01['PHIT'], k)
                
                hingle.plot_saturation(Rwa, m, a, n)
        # with st.beta_expander('Hingleplot'):
        #     phit_gt_01 = well_data[well_data['PHIT'] > 0.1]
    
        #     if not phit_gt_01.empty:
                
        #         st.set_option('deprecation.showPyplotGlobalUse', False)
        #         hingle = Hingle_plot((1/phit_gt_01['RESD'])^(1/m), phit_gt_01['PHIT'])
                
        #         hingle.plot_saturation(Rwa, m, a, n)
