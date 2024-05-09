import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class Pickett_plot():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sat_lines = []
        
    def plot_saturation(self, Rwa, m=2, a=1, n=2):
        plt.figure()
        plt.xlabel('Rt')
        plt.ylabel('Porosity')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(xmin=0.1, xmax=10000)
        
        sw = (1.0, 0.8, 0.6, 0.4, 0.2)
        phie = (0.01, 1)
        rt = np.zeros((len(sw), len(phie)))
                            
        for i in range(len(sw)):
            for j in range(len(phie)):
                rt_out = ((a * Rwa) / (sw[i] ** n) / (phie[j] ** m))
                rt[i, j] = rt_out      
        
        for i in range(len(sw)):
            line, = plt.plot(rt[i], phie, label=f'SW {int(sw[i]*100)}%')
            self.sat_lines.append(line)
            plt.legend(loc='best')
            plt.grid(which='both')
        
        plt.scatter(self.x, self.y, marker='.')
        st.pyplot()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go

# class Pickett_plot():
    
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.sat_lines = []
        
#     def plot_saturation(self, Rwa, m=2, a=1, n=2):
#         fig = go.Figure()
#         fig.update_layout(
#             xaxis_title='Rt',
#             yaxis_title='Porosity',
#             xaxis_type='log',
#             yaxis_type='log',
#             xaxis_range=[0.1, 10000],
#             yaxis_range=[0.01, 1]
#         )
        
#         sw = (1.0, 0.8, 0.6, 0.4, 0.2)
#         phie = (0.01, 1)
#         rt = np.zeros((len(sw), len(phie)))
        
#         for i in range(len(sw)):
#             for j in range(len(phie)):
#                 rt_out = ((a * Rwa) / (sw[i] ** n) / (phie[j] ** m))
#                 rt[i, j] = rt_out
                
#         for i in range(len(sw)):
#             fig.add_trace(go.Scatter(x=rt[i], y=phie, mode='lines', name=f'SW {int(sw[i]*100)}%'))
        
#         fig.add_trace(go.Scatter(x=self.x, y=self.y, mode='markers', name='Data'))
        
#         st.plotly_chart(fig)

# Usage example:
# well_data = pd.DataFrame({'RESD': np.random.uniform(0.1, 1000, 100), 'PHIT': np.random.uniform(0.1, 1, 100)})
# phit_gt_01 = well_data[well_data['PHIT'] > 0.1]
# pickett = Pickett_plot(phit_gt_01['RESD'], phit_gt_01['PHIT'])
# pickett.plot_saturation(Rwa=0.033, m=2, a=1, n=2)
