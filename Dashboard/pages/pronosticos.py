import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
from modules import util
from dash import callback
import base64
import io
from dash import Dash, dcc, html, Input, Output, no_update
import pandas as pd

dash.register_page(__name__)

layout = html.Div([
        # Left column
        html.Div([
            html.H1('Liverpool Human Analytics', className='title'),
            html.H3('Explora el Dashboard de Pronósticos de Liverpool', className='subtitle'),
            html.P('Ingresa al Dashboard de Human Analytics de Liverpool y accede a los pronósticos más precisos sobre la plantilla de empleados. Explora datos actualizados, selecciona sectores o periodos específicos, y toma decisiones informadas para mejorar la eficiencia en la gestión de recursos humanos.', className='description'),
            html.H4('Selecciona un sector', className='picker-description'),
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label': 'Todos', 'value': 'Todos'},
                    {'label': 'Liverpool', 'value': 'Liverpool'},
                    {'label': 'Suburbia', 'value': 'Suburbia'},
                    {'label': 'CeDis', 'value': 'CeDis'},
                    {'label': 'Otros', 'value': 'Otros'},
                ],
                value='Todos',
            ),
            html.H4('Carga un archivo CSV', className='picker-description'),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Subir archivo CSV', className='upload-button'),
                multiple=False
            ),
        ], className='left-column', style={'width': '30%'}),
        
        # Right column with charts
        html.Div([

            html.Div([
                html.Div([
                    html.Div(html.Img(src = 'assets/src/6.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Empleados en Muestra'), className='statistic-title'),
                    html.Div(html.A('0'), className='statistic-value', id = 'employees-total')
                ], className='statistics-box'),

                html.Div([
                    html.Div(html.Img(src = 'assets/src/7.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Empleados en Riesgo Alto'), className='statistic-title'),
                    html.Div(html.A('0'), className='statistic-value', id = 'high-risk-total')
                ], className='statistics-box'),

                html.Div([
                    html.Div(html.Img(src = 'assets/src/5.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Antigüedad promedio pronosticada'), className='statistic-title'),
                    html.Div(html.A('0 meses'), className='statistic-value', id = 'average-service')
                ], className='statistics-box')

            ], className= 'statistics-container'),
            
            html.Div([
               html.Div([
                     html.Div([

                        html.Div([
                            html.Div('Riesgos de renuncia por clasificación', className='risk-title-text'),
                            html.Hr(className='divider'),
                        ], className='risk-title'),

                        html.Div([
                            html.Div([
                                html.Div(html.A('Riesgo Alto'), className='risk-value')
                                ], className='risk-box'),
                            html.Div([html.A('0%')], className='risk', id='high-risk-percentage')
                        ], className= 'risk-boxes'),

                        html.Div([
                            html.Div([
                                html.Div(html.A('Riesgo Medio'), className='risk-value')
                                ], className='risk-box-1'),
                            html.Div([html.A('0%')], className='risk-1', id='medium-risk-percentage')
                        ], className= 'risk-boxes'),

                        html.Div([
                            html.Div([
                                html.Div(html.A('Riesgo Bajo'), className='risk-value')
                                ], className='risk-box-2'),
                            html.Div([html.A('0%')], className='risk-2', id='low-risk-percentage')
                        ], className= 'risk-boxes'),


                    ], className='risk-container'),

                    html.Div([

                        html.Div([
                            html.Div('Rangos de Antigüedad Pronosticada', className='risk-title-text'),
                            html.Hr(className='divider'),
                        ], className='risk-title'),

                        html.Div([
                            dcc.Graph(
                                id='service-prognose-groups', 
                                figure=util.plotServiceBars('Todos'),
                            ),
                        ], className='risk-chart')

                    ], className='risk-container'),

               ], className='risk-containers-line'),

               html.Div([
                    html.Div('Pronosticos de renuncias', className='chart-title'),
                    html.Hr(className='divider'),  # Horizontal line
                    dcc.Graph(
                        id='prognose-table',  # Replace with the ID of your graph
                        figure=util.getPredictionsTable('Todos', None),  # Initial value or default
                    ),
                ], className = 'chart-container'),
               
                # Add more charts here as needed
            ], className='charts-column')
        ], className='right-column', style={'width': '70%'}),
    ], className='two-column-layout')

@callback(
    Output('employees-total', 'children'),
    
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_total_employees_at_risk(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.atRisk(selected_value, df)
    return value

@callback(
    Output('high-risk-total', 'children'),
    
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_total_employees_at_high_risk(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.atHighRisk(selected_value, df)
    return value

@callback(
    Output('average-service', 'children'),
    
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_average_service(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.averageServicePrognosed(selected_value, df)
    return value

@callback(
    Output('high-risk-percentage', 'children'),
    
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_percentage_high_risk(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.percentageAtHighRisk(selected_value, df)
    return value

@callback(
    Output('medium-risk-percentage', 'children'),
    
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_percentage_medium_risk(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.percentageAtMediumhRisk(selected_value, df)
    return value

@callback(
    Output('low-risk-percentage', 'children'),
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_percentage_low_risk(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    value = util.percentageAtLowRisk(selected_value, df)
    return value

@callback(
    Output('service-prognose-groups', 'figure'),  # Replace 'resignation-graph' with your graph ID
    Input('dropdown', 'value'),
    
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_resignation_graph(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    # Call the util function with the selected value
    figure = util.plotServiceBars(selected_value, df)
    return figure

@callback(
    Output('prognose-table', 'figure'),
    Input('dropdown', 'value'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def update_uploaded_data(selected_value, contents):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    figure = util.getPredictionsTable(selected_value, df)

    return figure