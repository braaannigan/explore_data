#Data manipulation and analysis imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#Plotting imports
from bokeh.io import curdoc
from bokeh.models.glyphs import Text
from bokeh.layouts import row, widgetbox, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure


def lagged(df_original, lags = None, regression = True,
units = 'Units', pred_title = 'Predictor', resp_title = 'Response'):
    """Create a bokeh plot of lagged time series
    input: df_original = {'time','pred','resp'};
    regression = True means add the line of best fit and calculate r2;
    no output"""
    #Set axis limits
    if not lags:
        lags = np.arange(-36,36)
    lag = 0 #Initial lag
    #Create r2
    #df_r2 = pd.DataFrame(index = lags, columns = ['r2', 'r2_x', 'r2_y', 'alpha', 'beta','filled'])
    ax_lim = 1.1*np.max(np.abs(np.array([df_original['pred'],df_original['resp']])))
    x_plot = np.linspace(-ax_lim,ax_lim)

    def linear_regression(df, x_plot):
        """Calculate out the linear regression and r2 score"""
        #Instantiate the linear regression model
        model = LinearRegression()
        #Fit the model
        model.fit(df['pred'][:,np.newaxis],df['resp'])
        #Get the intercept
        alpha = model.intercept_
        #and the slope
        beta = model.coef_
        #Calculate the r2 score
        r2 = r2_score(df['resp'],model.predict(df['pred'][:,np.newaxis]))
        #Position for the r2 text annotation
        r2_position = [-ax_lim + 0.1*ax_lim, ax_lim - 0.1*ax_lim]
        r2_text = "R^2 = %02f" % r2
        return alpha, beta, r2,  r2_position, r2_text
    #Do linear regression for the initial lag
    if regression:
        alpha, beta, r2, r2_position, r2_text = linear_regression(df_original, x_plot)
        y_plot = alpha + beta*x_plot
    #Set the ColumnDataSources for the scatter plot
    s1_source = ColumnDataSource(data=dict(x=df_original['pred'], y=df_original['resp'])) #Scatter plot
    if regression:
        text_source = ColumnDataSource(dict(x=[r2_position[0]], y=[r2_position[1]], text=[r2_text])) #R2 value
        line_source = ColumnDataSource(data=dict(x=x_plot, y=y_plot)) #Regression line
    #Set the ColumnDataSources for the time series plot
    s2_source_pred = ColumnDataSource(data=dict(x=df_original['time'], y=df_original['pred'])) #Scatter plot
    s2_source_resp = ColumnDataSource(data=dict(x=df_original['time'], y=df_original['resp'])) #Scatter plot


    # Set up initial plot
    s1 = figure(plot_height=500, plot_width=500, title="Lagged scatter plot",
    tools="crosshair,pan,reset,save,wheel_zoom",
    x_range=[-ax_lim, ax_lim],
    y_range=[-ax_lim, ax_lim])

    s1.scatter('x', 'y', source=s1_source)
    s1.xaxis.axis_label = pred_title
    s1.yaxis.axis_label = resp_title
    if regression:
        s1.line('x', 'y', source = line_source, color = 'black')
        #Add the r2 text annotation
        glyph = Text(x="x", y="y", text="text", text_color="black")
        s1.add_glyph(text_source, glyph)

    #Set up time series plot
    s2 = figure(plot_height=500, plot_width=750, title="Lagged time series",
    tools="crosshair,pan,reset,save,wheel_zoom",toolbar_location='right',
    x_range=[df_original.time.values[0], df_original.time.values[-1]],
    y_range=[-ax_lim, ax_lim])

    s2.line('x', 'y', source = s2_source_pred, color = 'red',legend = pred_title)
    s2.line('x', 'y', source = s2_source_resp, color = 'blue',legend = resp_title)
    s2.yaxis.axis_label = units
    s2.xaxis.axis_label = 'Time'
    # Set up slider
    offset = Slider(title="Predictor leads by", value=0, start=-30, end=30, step=1)

    def update_data(attrname, old, new):
        # Get the new slider value
        lag = offset.value
        # Generate the new dataframe
        df = pd.DataFrame()
        if lag > 0:
            df['time'] = df_original['time'][lag:].values
            df['pred'] = df_original['pred'][:-lag].values
            df['resp'] = df_original['resp'][lag:].values
        elif lag == 0:
            df['time'] = df_original['time'].values
            df['pred'] = df_original['pred'].values
            df['resp'] = df_original['resp'].values
        else:
            df['time'] = df_original['time'][:lag].values
            df['pred'] = df_original['pred'][-lag:].values
            df['resp'] = df_original['resp'][:lag].values
        if regression:
            #Do linear regression for this lag
            alpha, beta, r2, r2_position, r2_text = linear_regression(df, x_plot)
            #Line of best fit for this lag
            y_plot = alpha + beta*x_plot
            #Update the line of best fit
            line_source.data = dict(x=x_plot, y=y_plot)
            #Update the r2 text annotation
            text_source.data = dict(x=[r2_position[0]], y=[r2_position[1]], text = [r2_text])

        #Update the scatter plot data
        s1_source.data = dict(x=df['pred'], y=df['resp'])
        #Update the predictor time series
        s2_source_pred.data = dict(x=df['time'],y=df['pred'])
        #Update the response time series
        s2_source_resp.data = dict(x=df['time'],y=df['resp'])

    #Make changes as the slider is changed
    offset.on_change('value', update_data)

    # Set up layouts and add to document
    inputs = widgetbox(offset) #Widgetbox only has a slider
    curdoc().add_root(gridplot([[s1,inputs],[s2,None]]))#Create the subplot grid
    curdoc().title = "Lagged" #Title in the web browser
