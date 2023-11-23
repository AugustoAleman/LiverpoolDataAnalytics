'''
 * Autor:           Octavio Augusto Aleman Esparza A01660702
 * Titulo:          util.py
 * Descripcion:     Archivo con funciones complementarias para crear Graficos con Plotly para Analitica de Datos
 * Fecha:           21.09.2023
 '''

### LIBRERIAS
import pandas as pd
import numpy as np
import requests

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.express as px

import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from datetime import datetime, date
from datetime import datetime, timedelta
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
        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]

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
        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]
  
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


def addStateInfo(df):
    file_name = os.path.join(os.path.dirname(__file__), '../data/CP Liverpool.csv')

    df_states = pd.read_csv(file_name)

    df_states = df_states.drop('Descubica', axis = 1)

    df_states['Estado'] = df_states['Estado'].str.capitalize()

    old_names = ['Ciudad de mexico', 
                 'Mexico', 
                 'Michoacan de ocampo', 
                 'Nuevo Leon',
                 'Queretaro',
                 'San luis potosi',
                 'Yucatan',
                 'Veracruz de ignacio de la llave',
                 'Nuevo leon',
                 'Baja california sur',
                 'Baja california',
                 'Quintana roo',
                 'Coahuila de zaragoza']
    
    new_names = ['Ciudad de México', 
                 'México', 
                 'Michoacán', 
                 'Nuevo León',
                 'Querétaro',
                 'San Luis Potosí',
                 'Yucatán',
                 'Veracruz',
                 'Nuevo León',
                 'Baja California Sur',
                 'Baja California',
                 'Quintana Roo',
                 'Coahuila']
    
    for key in range(0, len(old_names)):
        df_states['Estado'] =  df_states['Estado'].replace(old_names[key], new_names[key])

    df = df.merge(df_states, on = 'CP Trabajo', how = 'left')

    return df

df = loadCSV()

def resignedPerStateMap(area = 'Todos', start_date = None, end_date = None):

    df_aux = loadCSV()

    if not start_date == None and not end_date == None:
        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df_aux['Fecha Salida'] <= end_date)]

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['Descubica'].str.contains(area)]
        elif area == 'Otros':
            df_aux = df_aux[~df_aux['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]

    df_aux = addStateInfo(df_aux)

    repo_url = 'https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json'

    #Archivo GeoJSON
    mx_regions_geo = requests.get(repo_url).json()

    value_per_state = df_aux.groupby('Estado')['Nº pers.'].nunique()

    df = pd.DataFrame({'Estado': value_per_state.index.tolist(),
                      'Total': value_per_state.values.tolist()})
    
    df['Total'] = df['Total'].apply(lambda x: round((x / df['Total'].sum()) * 100, 2))

    fig = go.Figure(go.Choroplethmapbox(name='Mexico',
                                        geojson=mx_regions_geo,
                                        ids=df['Estado'],
                                        z=df['Total'],
                                        locations=df['Estado'],
                                        featureidkey='properties.name',
                                        colorscale=colors_extended[::-1],
                                        marker=dict(line=dict(color='black'),
                                                    opacity=0.6)))

    fig.update_layout(mapbox_style='open-street-map',
                    mapbox_zoom=3.5,
                    mapbox_center = {'lat': 24, 'lon': -102}
                    )

     #Configuración de título
    fig.update_layout(
            title = setTitle('Porcentaje de renuncias por estado (%)', f'Sector: {area}'),
            title_x=0.15,
            coloraxis_colorbar =dict (
                title = 'Porcentaje %'
            )
    )

    fig.layout.images = getLogo(0.02, 1.075, 0.22, 0.22)

    return fig

def resignedInPeriod(area = 'Todos', start_date = None, end_date = None):

    if not start_date == None and not end_date == None:
        df_aux = df

        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]

        if area == 'Todos':
            pass
        else:
            if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
                df_aux = df_aux[df_aux['Descubica'].str.contains(area)]
            elif area == 'Otros':
                df_aux = df_aux[~df_aux['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]

    if not start_date == None and not end_date == None:
        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]

        value = df_aux['Nº pers.'].nunique()

        return f'{value:,} personas'
    else:
        return 'Error'

def averageService(area, start_date, end_date):
    if not start_date == None and not end_date == None:
        df_aux = df

        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]

        if area == 'Todos':
            pass
        else:
            if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
                df_aux = df_aux[df_aux['Descubica'].str.contains(area)]
            elif area == 'Otros':
                df_aux = df_aux[~df_aux['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]

    if not start_date == None and not end_date == None:
        df_aux = df_aux[(df_aux['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)]

        print(df_aux['Nº pers.'].nunique())

        value = df_aux['Antigüedad'].mean()

        return f'{round(value, 1):,} años'
    else:
        return 'Error'
    
def turnoverRate(area = 'Todos', start_date = None, end_date = None):
    if not start_date == None and not end_date == None:

        demographic_active = os.path.join(os.path.dirname(__file__), '../data/Demograficos-Activos-Liverpool.csv')
        df_demographic_active = pd.read_csv(demographic_active)

        df_aux_res = df
        df_aux_act = df_demographic_active

        if area == 'Todos':
            pass
        else:
            if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
                df_aux_res = df_aux_res[df_aux_res['Descubica'].str.contains(area)]
                df_aux_act = df_aux_act[df_aux_act['Desc Ubicación'].str.contains(area)]
            elif area == 'Otros':
                df_aux_res = df_aux_res[~df_aux_res['Descubica'].str.contains('Liverpool|Suburbia|CeDis')]
                df_aux_act = df_aux_act[~df_aux_act['Desc Ubicación'].str.contains('Liverpool|Suburbia|CeDis')]
        
        # cast as date
        df_aux_act['Fecha Ingreso'] = pd.to_datetime(df_aux_act['Fecha Ingreso'], format='%Y-%m-%d')

        start_count = 0

        start_count += len(df_aux_act[(df_aux_act['Fecha Ingreso'] < start_date)])
        start_count += len(df_aux_res[(df_aux_res['Fecha Salida'])  > start_date])

        end_count = 0

        end_count += len(df_aux_act[(df_aux_act['Fecha Ingreso'] <= end_date)])
        end_count += len(df_aux_res[(df_aux_res['Fecha Salida'])  > end_date])

        resigned_in_period = len(df_aux_res[(df_aux_res['Fecha Salida'] >= start_date) & (df['Fecha Salida'] <= end_date)])

        value = (resigned_in_period / ((start_count + end_count) / 2)) * 100

        return f'{round(value, 1):,}%'

    else:
        return 'Error'
    
''' Modelos de ML '''

def loadModeldata():

    path = os.path.join(os.path.dirname(__file__), './model_saves/classification_model.joblib')

    classification_model = joblib.load(path)

    path = os.path.join(os.path.dirname(__file__), './model_saves/regression_model_.joblib')

    regression_model = joblib.load(path)

    return classification_model, regression_model

# Conversión de columnas categóricas a numericas
def dataToNumeric(df):

    df_categoric = df.select_dtypes(exclude= ['number'])

    label_encoder = LabelEncoder()
    for col in range (0, len(df_categoric.columns)):
        df[df_categoric.columns[col]] = label_encoder.fit_transform(df[df_categoric.columns[col]])

    return df

def identify_risk(df, model):
    
    model = model
            
    aux = df.copy()
    aux.drop(['genero', 'sindicato', 'edad ingreso', 'edad salida', 'id'], axis = 1, inplace = True)
    aux = dataToNumeric(aux)

    # Normalización
    n_colsX = aux.shape[1]-1
    X = aux.iloc[:,0:n_colsX]
    rescaledX = StandardScaler().fit_transform(X)
    newX    = pd.DataFrame(data=rescaledX,columns=X.columns)

    # Utilizar el modelo para predecir probabilidades de la clase positiva
    Y_probabilities = model.predict_proba(newX)[:, 1]

    # Definir umbrales de riesgo
    low_threshold = 0.3
    medium_threshold = 0.5

    # Asignar niveles de riesgo
    risk_levels = []

    for score in Y_probabilities:
        if score < low_threshold:
            risk_levels.append("Riesgo Bajo")
        elif low_threshold <= score < medium_threshold:
            risk_levels.append("Riesgo Medio")
        else:
            risk_levels.append("Riesgo Alto")

    # Agregar los niveles de riesgo al DataFrame
    df['Nivel de Riesgo'] = risk_levels

    return df

def identify_service(df, model):

    model = model

    # Colocar antiguedad como última columna
    class_column = df['antiguedad']
    df = df.drop('antiguedad', axis=1)
    df['antiguedad'] = class_column

    aux = df.copy()

    # Eliminación de columnas no usadas
    aux.drop(['clase', 'id', 'genero', 'eventual', 'sindicato'], axis = 1, inplace = True)

    # Conversión de datos categoricos
    n_colsX = aux.shape[1] - 1
    X = aux.iloc[:, 0:n_colsX]
    X = dataToNumeric(X)

    # Normalization
    rescaledX = StandardScaler().fit_transform(X)
    newX = pd.DataFrame(data=rescaledX, columns=X.columns)

    # Use the model to predict
    y_pred = model.predict(newX)

    # Add the predicted values to the DataFrame
    df['Antigüedad Pronosticada'] = y_pred
    df['Antigüedad Pronosticada'] = df['Antigüedad Pronosticada'].astype(float)

    # Conversión a meses
    df['Antigüedad Pronosticada'] = ((df['Antigüedad Pronosticada'] * 365.25) / 30)
    df['antiguedad'] = ((df['antiguedad'] * 365.25) / 30)

    # Redondear valores
    df['Antigüedad Pronosticada'] =  df['Antigüedad Pronosticada'].astype(int)
    df['antiguedad'] =  df['antiguedad'].astype(int)

    return df

def use_models(df):

    c_model, r_model = loadModeldata()

    if df is None:
        path = os.path.join(os.path.dirname(__file__), '../data/model/sample_1.csv')
        model_df = pd.read_csv(path)
        aux = model_df.copy()
    else:
        aux = df.copy()

    aux = identify_risk(aux, c_model)

    aux = identify_service(aux, r_model)

    return aux

model_df = use_models(None)

def atRisk(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    return f'{round(len(df_aux), 0):,}'

def atHighRisk(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    df_aux = df_aux[df_aux['Nivel de Riesgo'] == 'Riesgo Alto']

    return f'{round(len(df_aux), 0):,}'

def averageServicePrognosed(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    aux = df_aux['Antigüedad Pronosticada'].mean()
    aux = int(aux)

    return f'{aux:,} meses'

def percentageAtHighRisk(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    df_aux = df_aux[df_aux['Nivel de Riesgo'] == 'Riesgo Alto']

    return f'{round(((len(df_aux) / len(df)) * 100), 2):,}%'

def percentageAtMediumhRisk(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    df_aux = df_aux[df_aux['Nivel de Riesgo'] == 'Riesgo Medio']

    return f'{round(((len(df_aux) / len(df)) * 100), 2):,}%'
   
def percentageAtLowRisk(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    df_aux = df_aux[df_aux['Nivel de Riesgo'] == 'Riesgo Bajo']

    return f'{round(((len(df_aux) / len(df)) * 100), 2):,}%'

# Grafico de Barras
def plotBars(df, 
             column, 
             data, 
             name = '', 
             title = 'Titulo', 
             subtitle = '',
             x_label = '', 
             y_label = '',
             orientation = 'v'):
    
    fig = go.Figure()

    bars = []

    color_key = 2

    patterns = [ '', '/', '\\', 'x', '-', '|', '+', '.']

    for key in range(0, len(data)):
         fig.add_trace(go.Bar(x = df[data[key]] if orientation == 'horizontal' else df[column],
                     y = df[column] if orientation == 'horizontal' else df[data[key]],
                     marker = dict(color = colors[key + color_key]),
                     name = data[key],
                     marker_pattern_shape = patterns[key],
                     orientation = 'h' if orientation == 'horizontal' else 'v'))
         
         color_key += 1

    fig.update_xaxes(showline=True, 
                     linewidth=2, 
                     linecolor='black', 
                     showgrid=False,
                     tickangle=45,
                     title_text = y_label if orientation == 'horizontal' else x_label)
    
    fig.update_yaxes(showline=True, 
                     linewidth=2, 
                     linecolor='black', 
                     showgrid=False,
                     title_text = x_label if orientation == 'horizontal' else y_label)

    fig.update_layout(height = 260,
                      width = 420,
                      #title_text= setTitle(title, subtitle + f'- Dataset: {name}'),
                      showlegend = False,
                      plot_bgcolor='white',
                      )
    
    return fig

def plotServiceBars(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]
    
    # Crear un diccionario para definir los patrones y sus rangos
    patrones = {
        '< 6 meses': (0, 6),
        '6 -12 meses': (6, 12),
        '12 - 24 meses': (12, 24),
        '> 24 meses': (24, 60)
    }

    # Inicializar un diccionario para almacenar el conteo por grupo
    conteo_por_grupo = {'Grupo': [], 'Conteo': []}

    # Iterar sobre los patrones y contar las coincidencias en cada rango
    for grupo, rango in patrones.items():
        filtro = (df_aux['Antigüedad Pronosticada'] > rango[0]) & (df_aux['Antigüedad Pronosticada'] <= rango[1])
        conteo = df_aux[filtro].shape[0]
        conteo_por_grupo['Grupo'].append(grupo)
        conteo_por_grupo['Conteo'].append(conteo)

    # Crear el dataframe auxiliar
    aux_df = pd.DataFrame(conteo_por_grupo)

 
    values = aux_df['Conteo'].iloc[0:4].tolist()

    fig = go.Figure(data=[go.Pie(labels=list(patrones.keys()),
                             values=values,
                             pull=[0, 0, 0, 0.2])])  # Ajusta el valor del cuarto elemento según sea necesario

    fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=12,
                    marker=dict(colors=colors, line=dict(color='#000000', width=2)))

    fig.update_layout(
        showlegend=False,  # No mostrar leyendas
        height=330,
        width=440
    )

    return fig


def getPredictionsTable(area = 'Todos', df = None):

    if df is None:
        df_aux = model_df.copy()
    else:
        df_aux = use_models(df)

    if area == 'Todos':
        pass
    else:
        if area == 'Liverpool' or area == 'Suburbia' or  area == 'CeDis':
            df_aux = df_aux[df_aux['area empresarial'].str.contains(area)]
        elif area == 'Otros':
                df_aux = df_aux[~df_aux['area empresarial'].str.contains('Liverpool|Suburbia|CeDis')]

    # Convierte 'antiguedad' en un formato de fecha
    df_aux['Fecha Ingreso'] = datetime.now() - pd.to_timedelta(df_aux['antiguedad'], unit='D')

    # Calcula la 'Fecha Salida' sumando 'Antigüedad Pronosticada' a 'Fecha Ingreso'
    df_aux['Fecha Salida'] = df_aux.apply(lambda row: row['Fecha Ingreso'] + pd.DateOffset(months=row['Antigüedad Pronosticada']), axis=1)

    # Aplica la lógica condicional para determinar el contenido de la columna 'Fecha Salida'
    df_aux['Fecha Salida'] = df_aux.apply(lambda row: f"{row['Fecha Salida'].strftime('%m/%Y')}" if row['Fecha Salida'] > datetime.now() else 'Alcanzada', axis=1)

    # Elimina la columna temporal 'Fecha Ingreso' si no la necesitas en el resultado final
    df_aux = df_aux.drop(columns=['Fecha Ingreso'])
    

    fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>ID</b>','<b>Área Empresarial</b>','<b>Nivel de Riesgo</b>','<b>Antigüedad Pronosticada (meses)</b>', '<b>Fecha estimada de Salida</b>'],
        line_color='white',
        fill_color=colors[2],
        align=['left','center'],
        font=dict(color='white', size=12)
    ),
    cells=dict(
        values=[
        df_aux['id'],
        df_aux['area empresarial'],
        df_aux['Nivel de Riesgo'],
        df_aux['Antigüedad Pronosticada'],
        df_aux['Fecha Salida']],
        line_color='white',
        # 2-D list of colors for alternating rows
        fill_color = [['white',colors[5],'white', colors[5],'white', colors[5]]*1000],
        align = ['left', 'center'],
        font = dict(color = 'darkslategray', size = 11)
        ))
    ])

    return fig

