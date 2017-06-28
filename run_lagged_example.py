#This script shows an example of how to use the interactive months-lagged
#correlation function `lagged` in slides.py
#To execute this script run `bokeh serve --show lagged_correlations.py in
#the *bash* command line.  Nothing will appear if its run in the python
#interpreter

#Imports for managing data
import numpy as np
import pandas as pd
#Import the plotting function
from slides import lagged

#Create months-lagged datasets
months = np.linspace(0,120,120) #Ten years in `months'
freq = 2.*np.pi/18 #One cycle every 18 months
pred = np.cos(freq*months) #Predictor months series
y_in = np.cos(freq*months + 2*np.pi*(0.33)) + (np.random.rand(len(months))-0.5) #Response months series lagged by 6 months
#Create the input dataframe
df = pd.DataFrame({'time':months, 'pred':pred, 'resp':y_in})

lagged(df)
