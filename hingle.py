import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
class Hingle_plot():
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sat_lines = []
        
    def plot_saturation(self, Rwa, m=2, a=1, n=2):
        plt.figure()
        plt.ylabel('(1/Rt)/(1/m)')
        plt.xlabel('Porosity')
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