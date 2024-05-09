# cd .\GUI
# python -m streamlit run  app.py


import streamlit as st
st.set_page_config(layout="wide", page_title='LAS Explorer v.0.1')

from load_css import local_css
import lasio
import missingno as mno
import numpy as np
import pandas as pd
# Local Imports
import home
import raw_data
import new_data
import plotting
import model
import header
import missingdata

from io import StringIO

local_css("style.css")


@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            str_io = StringIO(bytes_data.decode('Windows-1252'))
            las_file = lasio.read(str_io)
            well_data = las_file.df()
            well_data['DEPTH'] = well_data.index
            well = well_data.copy()
            well.dropna(inplace=True)

            if 'GR_CORR' in well.columns:
                well.drop('GR', axis=1, inplace=True)
                well.rename(columns={'GR_CORR':'GR'}, inplace=True)

            if 'RESD_CORR' in well.columns:
                well.drop('RESD', axis=1, inplace=True)
                well.rename(columns={'RESD_CORR':'RESD'}, inplace=True)
                
            if 'DENS_CORR' in well.columns:
                well.drop('DENS', axis=1, inplace=True)
                well.rename(columns={'DENS_CORR':'DENS'}, inplace=True)
            # GRmin=55
            # GRmax=150
            GRmin=well["GR"].min()
            GRmax=well["GR"].max()
            well=well.loc[(well['GR'] >= GRmin) & (well['GR'] <= GRmax)]
            
            # DTS=Vp  unit = m/s
            well["Vp"]=304800/well["DTC"]
            # DTS=Vs   unit = m/s
            well["Vs"]=304800/well["DTS"]
            # K=Bulk Modulus  unit = GPa
            well["K"]=well["DENS"]*(well["Vp"]**2-(4/3)*well["Vs"]**2)
            well["K"]=well["K"]/1000000   
            # G=Shear Modulus  unit = GPa
            # well["G"]=well["DENS"]*well["Vs"]**2
            # well["G"]=well["G"]/1000000 
            # # Vp/Vs
            # well["Vp/Vs"]=well["Vp"]/well["Vs"]
            # # Ve=extensional velocity
            # well["Ve"]=well["Vs"]*np.sqrt((3*(well["Vp/Vs"])**2-4)/(well["Vp/Vs"]**2-1))
            # # E=Youngs modulus  unit = GPa
            # well["E"]=well["DENS"]*well["Ve"]**2
            # well["E"]=well["E"]/1000000 
            # # mu=Poisson ratio
            # well["mu"]=(well["Vp/Vs"]**2-2)/(2*(well["Vp/Vs"]**2-1))
            # Vshale = volume of shale
            well["Vshale"]=(well["GR"]-GRmin)/(GRmax-GRmin)

        except UnicodeDecodeError as e:
            st.error(f"error loading log.las: {e}")
    else:
        las_file = None
        well_data = None
        well = None

    return las_file, well_data , well


#TODO
def missing_data():
    st.title('Missing Data')
    missing_data = well_data.copy()
    missing = px.area(well_data, x='DEPTH', y='DT')
    st.plotly_chart(missing)

# Sidebar Options & File Uplaod
las_file=None
st.sidebar.write('# LAS Data Explorer')
st.sidebar.write('To begin using the app, load your LAS file using the file upload option below.')

uploadedfile = st.sidebar.file_uploader(' ', type=['.las'])
las_file, well_data, well= load_data(uploadedfile)

# new_well_data = well_data
# new_well_data.dropna(inplace=True)

if las_file:
    st.sidebar.success('File Uploaded Successfully')
    st.sidebar.write(f'<b>Well Name</b>: {las_file.well.WELL.value}',unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title('Navigation')
# options = st.sidebar.radio('Select a page:', 
#     ['Home', 'Header Information', 'Data Information', 'Data Visualisation', 'Missing Data Visualisation'])
options = st.sidebar.radio('Select a page:', 
    ['Home', 'Header Information', 'Data Information','Modified Data Information', 'Data Visualisation','Elastic Bound Model'])
if options == 'Home':
    home.home()
elif options == 'Header Information':
    header.header(las_file)
elif options == 'Data Information':
    raw_data.raw_data(las_file, well_data)
elif options == 'Modified Data Information':
    new_data.new_data(las_file, well)
elif options == 'Data Visualisation':
    plotting.plot(las_file, well_data)
elif options == 'Elastic Bound Model':
    model.model(las_file, well)

# elif options == 'Missing Data Visualisation':
#     missingdata.missing(las_file, well_data)