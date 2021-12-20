import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def acc_mortal(dfdata, col_group, target):
    '''
    dfdata será nuestro dataset
    col_group la columna por la que queremos agrupar
    target va a ser c_sev, puesto que nos interesa saber si hubo fallecidos
     - suma todas las repeticiones y llámalas accidentes
     - resetea el índice
     - le añadimos el target que será fallecidos
     - calculamos la mortalidad
    '''    
    df1 = pd.DataFrame(dfdata[col_group].value_counts())
    df1.reset_index(inplace= True)
    df1.rename(columns={col_group : 'accidentes', 'index': col_group}, inplace=True)
    df1 = df1.sort_values(by=[col_group])
    df1
    
    df2 = dfdata.groupby(col_group)[target].sum().reset_index()
    df2 = df1.merge(df2)
    df2['pct_mortales'] = (df2[target] * 100 ) / df2['accidentes']
    df2['acc_ponderado'] = ((df2['accidentes'] / df2['accidentes'].sum()) * 100)
    df2['muertes_ponderado'] = ((df2['acc_mortal'] / df2['acc_mortal'].sum()) * 100)

    df2[['accidentes', target]] = df2[['accidentes', target]].astype(int)
    return(df2)

''''
Intentaremos graficar los accidentes horarios de forma circular:
va a coger x y va a conseguir los radianes. Int porque nuestra 
variable hora es tipo object, para que la transforme. 2 * pi * h/24
con ese int para sacar los radianes
'''
def get_radian(x):
    h = int(x)
    return 2 * np.pi * (h)/24

def plot_hours(df, time_col, plot_col, plot_title= None):
    """
    Gráfico polar por horas
    df: va a ser df_horas
    time_col: nombre columna temporal (24h)
    plot_col: nombre de la columna que contamos y representamos
    plot_title: título del plot
    """
    
    fig = plt.figure(figsize=(8, 8))
    ts = df[time_col]
    ts = ts.apply(get_radian)
    ax = plt.subplot(111, projection= 'polar')
    ax.bar(ts, df[plot_col], width=0.25, alpha=0.3, color= 'red')
    
    # Dirección clockwise
    ax.set_theta_direction(-1)
    # El 0 van a ser las 12 AM:
    ax.set_theta_offset(np.pi / 2)
    # Le decimos los ticks de nuestra circunferencia
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    
    # Nombramos cada tick
    ticks = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM', '5 AM', '6 AM', '7 AM',
             '8 AM', '9 AM', '10 AM', '11 AM', '12 PM', '1 PM', '2 PM', '3 PM',
             '4 PM', '5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM']
    
    ax.set_xticklabels(ticks)
    # suppress the radial labels
    plt.setp(ax.get_yticklabels(), visible=False)
    # Radios: máximo de la longitud de nuestra columna.
    plt.ylim(0, max(df[plot_col]))
    plt.title(plot_title, pad=30, fontsize = 15)
    plt.show()