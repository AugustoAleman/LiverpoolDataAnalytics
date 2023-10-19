'''
 * Autor:           Octavio Augusto Aleman Esparza A01660702
 * Titulo:          util.py
 * Descripcion:     Archivo con funciones complementarias para crear Graficos con Plotly para Analitica de Datos
 * Fecha:           21.09.2023
 '''

### LIBRERIAS
import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.express as px

from datetime import datetime, date
import calendar

import re
import os

import warnings
warnings.filterwarnings('ignore')

### DEFINICION VARIABLES GLOBALES

# Paleta de colores
colors = ['#CD137A',
          '#E43A93',
          '#E860A1',
          '#EB85B0',
          '#EEAABF',
          '#F2D0CD',
          '#F5F5DC']

colors_extended = [ '#CD137A', 
                   '#D12A84', 
                   '#D5408E', 
                   '#D95797', 
                   '#DD6DA1', 
                   '#E184AB', 
                   '#E59BB5', 
                   '#E9B1BF', 
                   '#EDC8C8', 
                   '#F1DED2', 
                   '#F5F5DC'
]

title_colors = {'Title': '#CD137A',
                'Subtitle': '#717F94'}

# Logo Liverpool

def getLogo(x, y, sizex, sizey):
    logo = [dict(
        source='../assets/src/logo-pequenio.png',
        xref='paper', yref='paper',
        x=x, 
        y=y,
        sizex=sizex, 
        sizey=sizey,
        xanchor='center', yanchor='bottom'
      )]
    
    return logo

def setTitle(title = 'Titulo', subtitle = ''):
    new_title = f'<span style=\'font-size: 18px; font-family: Tahoma, sans-serif;  color: {title_colors["Title"]};\'>{title}<br><span style=\'font-size: 14px;color: {title_colors["Subtitle"]};\'><i>{subtitle}</i></span>'
    return new_title

### FUNCIONES DE EXPLORACION

# Establecer Index específico
def setCustomIndex(df, column = 0):
    df.set_index(df.columns[column], inplace=True)
    df.index.name = 'index'

    return df

# Mostrar Dataframe
def showDF(df, name = ''):
    print(f'Las primeras 5 entradas del Dataframe {name} son:')
    return df.head(5)

# Dimensiones del Dataframe
def dimensions(df, name = ''):
    n_rows, n_cols = df.shape
    print(f'El Dataset {name} cuenta con {n_cols} columnas y {n_rows} filas.')

# Checar el tipo de Datos
def dataTypes(df, name = ''):
    print(f'Los tipos de datos del dataset {name} son:')
    return df.info()

### FUNCIONES DE LIMPIEZA Y PREPROCESAMIENTO

# Resumen estadístico del Dataframe
def summaryStats(df, name = ''):
    print(f'El resumen estadístico de los valores númericos del Dataframe {name} es:')
    return df.describe(include='number')

# Valores faltantes por Dataframe
def nullValues(df, name = ''):
    print(f'Valores nulos por columna del Dataframe {name}:')
    return df.isnull().sum()

# Filas Duplicadas por Dataframe
def isDuplicated(df, name = ''):
    duplicated_rows = df.duplicated().sum()
    print(f'En el Dataframe {name} se encontraron un total de {duplicated_rows} filas duplicadas.')


# Conversion de Tipo de Datos
def casting(df, column, type):
    if type in ('int', 'bool', 'object', 'category', 'str'):
        df[column] = df[column].astype(type)
    elif type == 'float':
        df[column] = df[column].str.replace(',', '.')
        df[column] = df[column].astype(type)
    elif type == 'date':
        date_formats = [
            r'\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}',
            r'\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}:\d{1,2}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{4}/\d{1,2}/\d{1,2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
        ]
    
        formats = [ '%Y/%m/%d %H:%M:%S',
                    '%Y-%m-%d %H:%M:%S', 
                    '%Y-%m-%d', 
                    '%Y/%m/%d', 
                    '%d/%m/%Y']

        pattern = -1

        while pattern < 0:
            date_sample = df[column].sample(1).values
            for key in range(0, len(date_formats)):
                if re.match(date_formats[key], str(date_sample[0])):
                    pattern = key
                    break
        
        df[column] = pd.to_datetime(df[column], format = formats[pattern], errors = 'coerce')

        
    new_type = df[column].dtypes
    print(f'El tipo de la columna \'{column}\' fue cambiado exitosamente a {new_type}.')

    return df

# Manejo de Outliers - Metodo IQR
def iqr_transform(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    #   IQR
    IQR = Q3 - Q1

    #   Limite inferior
    LC = Q1 - (1.5 * IQR)

    #   Limite superior
    UC = Q3 + (1.5 * IQR)

    df = df[(df[column] > LC) & (df[column] < UC)]

    return df

#### DATAFRAME DEMOGRAFICOS RENUNCIAS

# Definicion de ruta de archivos

file_path = '../data/'

# Demografico renuncias

#Carga de archivo
def loadCSV():
    demographic_resigned = os.path.join(os.path.dirname(__file__), '../data/Demograficos-Renuncias-Liverpool.csv')
    df_demographic_resigned = pd.read_csv(demographic_resigned)
    df_demographic_resigned = setCustomIndex(df_demographic_resigned, 0)

    #Exploracion
    showDF(df_demographic_resigned, 'Bajas Demográfico')
    dimensions(df_demographic_resigned, 'Bajas Demográfico')
    dataTypes(df_demographic_resigned, 'Bajas Demográfico')

    #Limpieza y Pre-Procsamiento

    # Conversión de Tipo de Datos
    # Nº pers. a objeto
    df_demographic_resigned = casting(df_demographic_resigned, 'Nº pers.', 'object')

    # Fecha nacimiento a fecha
    df_demographic_resigned = casting(df_demographic_resigned, 'Fecha nacimiento', 'date')

    # Fecha ingreso a fecha
    df_demographic_resigned = casting(df_demographic_resigned, 'Fecha ingreso', 'date')

    # CP Trabajo a objeto
    df_demographic_resigned = casting(df_demographic_resigned, 'CP Trabajo', 'object')

    # Ubicación a objeto
    df_demographic_resigned = casting(df_demographic_resigned, 'Ubicación', 'object')

    # Fecha Salida a fecha
    df_demographic_resigned = casting(df_demographic_resigned, 'Fecha Salida', 'date')

    # Año Salida a objeto
    df_demographic_resigned = casting(df_demographic_resigned, 'Año salida', 'object')

    # Función a objeto
    df_demographic_resigned = casting(df_demographic_resigned, 'Función', 'object')

    #Valores faltantes
    nullValues(df_demographic_resigned, 'Bajas Demográfico')

    #Valores repetidos
    isDuplicated(df_demographic_resigned, 'Bajas Demográfico')

    #Resumen estadistico
    summaryStats(df_demographic_resigned, 'Bajas Demográfico')

    return df_demographic_resigned

# Dataset

df = loadCSV()


def resignationOverMonnths(area = 'Todos', start_date = None, end_date = None):
    df_aux = df

    print('start date:', start_date)
    print('end date:', end_date)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['Descubica'].str.contains(area)]
        elif area == 'Otros':
            df_aux = df_aux[~df_aux['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]

    if not start_date == None and not end_date == None:
        df_aux = df_aux[(df_aux['Fecha ingreso'] >= start_date) & (df['Fecha Salida'] <= end_date)]

    # Crear columna col el mes de la salida
    df_aux['Mes Salida'] =  pd.DatetimeIndex(df_aux['Fecha Salida']).month

    # Pasar el mes de salida a formato en letras
    df_aux['Mes Salida'] = df_aux['Mes Salida'].apply(lambda x: calendar.month_abbr[x])

    # Concatenar en una nueva columna mes y año de salida
    df_aux['Mes-Año'] = df_aux['Mes Salida'] + '-' + df_aux['Año salida'].astype(str)

    # Agrupar las coincidencias por mes y anio de salida de acuerdo al id de trabajador
    top_months_resignation = df_aux.groupby('Mes-Año')['Nº pers.'].nunique()

    # Convertir el índice 'Mes-Año' en formato datetime para permitir el ordenamiento
    top_months_resignation.index = pd.to_datetime(top_months_resignation.index, format='%b-%Y')

    # Ordenar la serie por fechas
    top_months_resignation = top_months_resignation.sort_index()

    #   Plot en grafica tipo stocks
    fig = px.line(df,
                x = top_months_resignation.index,
                y = top_months_resignation)

    # Configurar título y etiquetas
    fig.update_layout(
        title=setTitle('Bajas Por Meses', f'Sector Empresarial: {area}'),
        xaxis_title="Fecha",
        yaxis_title="Bajas",
        title_x=0.175,
        plot_bgcolor="white",
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black'),
    )

    fig.update_traces(line=dict(color = colors[0]))

    fig.layout.images = getLogo(0.05, 1.0, 0.195, 0.195)

    #   Para solo mostrar enero y Julio
    fig.update_xaxes(
        tickmode="auto",
        dtick="M1",
        tickformat="%b\n%Y"
    )

    return fig

# Garfico de pastel
def plotPie(labels, 
            values, 
            pull, 
            name, 
            title = 'Titulo', 
            subtitle = ''):
    
    fig = go.Figure(data=go.Pie(
        labels = labels,
        values = values,
        marker_colors=colors,
        pull = pull
    ))
 
    fig.update_traces(hoverinfo='label+percent', 
                      textfont_size=13, 
                      textinfo ='label+percent',
                      marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))
 
    fig.update_layout(
        title = setTitle(title, subtitle),
        title_x=0.18,
    )

    fig.layout.images = getLogo(-0.2, 1.068, 0.21, 0.21)

    return fig

def resignedPerArea(start_date = None, end_date = None):
  
  df_aux = df

  if not start_date == None and not end_date == None:
        df_aux = df_aux[(df_aux['Fecha ingreso'] >= start_date) & (df['Fecha Salida'] <= end_date)]
  
  # DEMOGRAFICOS
  # Separacion de Dataframes por Area empresarial
  df_liverpool = df_aux[df_aux['Descubica'].str.contains('Liverpool')]

  df_suburbia = df_aux[df_aux['Descubica'].str.contains('Suburbia')]

  df_cedis = df_aux[df_aux['Descubica'].str.contains('CeDis')]

  # Crear un DataFrame para los datos que no sean ninguna de las ubicaciones anteriores
  df_others = df_aux[~df_aux['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]

  # Agregar a un arreglo los valores de renuncias por area empresarial
  resigned_per_location = [len(df_liverpool),
                          len(df_suburbia),
                          len(df_cedis),
                          len(df_others)]

  labels = ['Liverpool', 'Suburbia', 'Centros de<br>Distribución', 'Otros']

  fig = plotPie(labels,
        resigned_per_location,
        [0, 0.2, 0, 0],
        'Demográficos de Renuncias',
        'Renuncias por Área Empresarial',
        'en proporción al total de bajas')

  return fig

df = loadCSV()

def areasOverYears(df = df):

  # Crear columna col el mes de la salida
  df['Mes Salida'] =  pd.DatetimeIndex(df['Fecha Salida']).month

  # Pasar el mes de salida a formato en letras
  df['Mes Salida'] = df['Mes Salida'].apply(lambda x: calendar.month_abbr[x])

  # Concatenar en una nueva columna mes y año de salida
  df['Mes-Año'] = df['Mes Salida'] + '-' + df['Año salida'].astype(str)

  # DEMOGRAFICOS
  # Separacion de Dataframes por Area empresarial
  df_liverpool = df[df['Descubica'].str.contains('Liverpool')]

  df_suburbia = df[df['Descubica'].str.contains('Suburbia')]

  df_cedis = df[df['Descubica'].str.contains('CeDis')]

  # Crear un DataFrame para los datos que no sean ninguna de las ubicaciones anteriores
  df_others = df[~df['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]


  title = 'Bajas por Año<br><i>distribuidas por Área Empresarial'
  labels = ['Liverpool', 'Suburbia', 'CeDis', 'Otros']

  dfs = [df_liverpool, df_suburbia, df_cedis, df_others]

  mode_size = [12, 10, 8, 8]
  line_size = [4, 3, 2, 2]

  # Crear un rango de fechas que cubra todos los meses presentes en los datos
  date_range = df.groupby('Mes-Año')['Nº pers.'].nunique()
  date_range.index = pd.to_datetime(date_range.index, format='%b-%Y')
  date_range = date_range.sort_index()
  date_range = date_range.index.tolist()

  total_resigned_year = []

  for year in range(2019, 2023):
      key = len(df[df['Fecha Salida'].dt.year == year])
      total_resigned_year.append(key)

  x_data = np.vstack((np.arange(2019, 2023),) * 4)

  # Inicializa una lista vacía para almacenar los resultados
  y_data = []

  # Itera a través de los DataFrames en dfs
  i = 0
  for df in dfs:
      aux = []
      for year in range(2019, 2023):
          count = round(((len(df[df['Fecha Salida'].dt.year == year]) / total_resigned_year[i]) * 100), 1)
          aux.append(count)
          i += 1
      y_data.append(aux)
      i = 0

  fig = go.Figure()

  j = 0
  for i in range(0, 4):
      fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines',
          name=labels[i],
          line=dict(color=colors_extended[j], width=line_size[i]),
          connectgaps=True,
      ))

      fig.add_trace(go.Scatter(
          x = [x_data[i][0], x_data[i][-1]],
          y = [y_data[i][0], y_data[i][-1]],
          mode = 'markers',
          marker=dict(color=colors_extended[j], size=mode_size[i])
      ))
      j += 2

  fig.update_layout(
      xaxis=dict(
          showline=True,
          showgrid=False,
          showticklabels=True,
          linecolor='rgb(204, 204, 204)',
          linewidth=2,
          ticks='outside',
          tickfont=dict(
              family='Arial',
              size=12,
              color='rgb(82, 82, 82)',
          ),
          tickmode='array',
          tickvals=x_data[0],
          ticktext=[str(year) for year in x_data[0]],
      ),
      yaxis=dict(
          showgrid=False,
          zeroline=False,
          showline=False,
          showticklabels=False,
      ),
      margin=dict(
          autoexpand=False,
          l=100,
          r=20,
          t=110,
      ),
      showlegend=False,
      plot_bgcolor='white'
  )

  annotations = []

  # Adding labels
  for y_trace, label, color in zip(y_data, labels, colors):
      # labeling the left_side of the plot
      annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                    xanchor='right', yanchor='middle',
                                    text=label + ' {}%'.format(y_trace[0]),
                                    font=dict(family='Arial',
                                              size=16),
                                    showarrow=False))
      # labeling the right_side of the plot
      annotations.append(dict(xref='paper', x=0.95, y=y_trace[3],
                                    xanchor='left', yanchor='middle',
                                    text='{}%'.format(y_trace[3]),
                                    font=dict(family='Arial',
                                              size=16),
                                    showarrow=False))
  # Title
  annotations.append(dict(xref='paper', yref='paper', x=0.1, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text=setTitle('Porcentaje de Bajas', 'Dividido por Área Empresarial'),
                                font=dict(family='Arial',
                                          size=20),
                                showarrow=False))

  fig.layout.images =getLogo(0.02, 1.05, 0.25, 0.25)

  fig.update_layout(annotations=annotations)

  return fig

