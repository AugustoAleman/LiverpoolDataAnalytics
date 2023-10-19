import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
from modules import util
from dash import callback

dash.register_page(__name__, path='/')

layout = html.Div([
        # Left column
        html.Div([
            html.H1('Liverpool Human Analytics', className='title'),
            html.H3('Bienvenido al Dashboard de Human Analytics de Liverpool', className='subtitle'),
            html.P('Explora los datos más recientes de Human Analytics dentro de Liverpool. Selecciona si deseas visualizar un sector o periodo específicos.', className='description'),
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
            html.H4('Selecciona un periodo', className='picker-description'),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=datetime(2019, 1, 1),
                end_date=datetime(2023, 1, 31),
            )
        ], className='left-column', style={'width': '30%'}),
        
        # Right column with charts
        html.Div([

            html.Div([
                html.Div([
                    html.Div(html.Img(src = 'assets/src/7.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Tasa de rotación'), className='statistic-title'),
                    html.Div(html.A('57.6%'), className='statistic-value', id = 'turnover-rate')
                ], className='statistics-box'),

                html.Div([
                    html.Div(html.Img(src = 'assets/src/6.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Bajas en periodo'), className='statistic-title'),
                    html.Div(html.A('5,600'), className='statistic-value', id = 'resigned-in-period')
                ], className='statistics-box'),

                html.Div([
                    html.Div(html.Img(src = 'assets/src/5.png', className='statistic-image'), className='statistic-image-container'),
                    html.Div(html.A('Antigüedad promedio p/ empleado'), className='statistic-title'),
                    html.Div(html.A('6.67 meses'), className='statistic-value', id = 'service-in-period')
                ], className='statistics-box')

            ], className= 'statistics-container'),
            
            html.Div([
                html.Div([
                    html.Div('Tendencia de renuncias', className='chart-title'),
                    html.Hr(className='divider'),  # Horizontal line
                    dcc.Graph(
                        id='resignation-graph',  # Replace with the ID of your graph
                        figure=util.resignationOverMonnths('Todos', None, None),  # Initial value or default
                    ),
                ], className = 'chart-container'),
                html.Div([
                    html.Div('Renuncias por estado', className='chart-title'),
                    html.Hr(className='divider'),  # Horizontal line
                    dcc.Graph(
                        id='resignation-map',  # Replace with the ID of your graph
                        figure=util.resignedPerStateMap('Todos', None, None),  # Initial value or default
                    ),
                ], className = 'chart-container'),
                html.Div([
                    html.Div('Renuncias en Sectores Estrátegicos', className='chart-title'),
                    html.Hr(className='divider'),
                    dcc.Graph(
                        id='resignation-per-area-graph', 
                        figure=util.resignedPerArea(None, None),
                    ),
                ], className = 'chart-container'),
                html.Div([
                    html.Div('Renuncias por Sector en el tiempo', className='chart-title'),
                    html.Hr(className='divider'),
                    dcc.Graph(
                        figure=util.areasOverYears(),
                    ),
                ], className = 'chart-container'),
                # Add more charts here as needed
            ], className='charts-column')
        ], className='right-column', style={'width': '70%'}),
    ], className='two-column-layout')

@callback(
    Output('resignation-graph', 'figure'),  # Replace 'resignation-graph' with your graph ID
    Input('dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_resignation_graph(selected_value, start_date, end_date):
    # Call the util function with the selected value
    figure = util.resignationOverMonnths(selected_value, start_date, end_date)
    return figure

@callback(
    Output('resignation-map', 'figure'),  # Replace 'resignation-graph' with your graph ID
    Input('dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_resignation_graph(selected_value, start_date, end_date):
    # Call the util function with the selected value
    figure = util.resignedPerStateMap(selected_value, start_date, end_date)
    return figure

@callback(
    Output('resignation-per-area-graph', 'figure'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_resignation_per_area_graph(start_date, end_date):
    # Call the util function with the selected value
    figure = util.resignedPerArea(start_date, end_date)
    return figure

@callback(
    Output('resigned-in-period', 'children'),
    Input('dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_resigned_in_period(selected_value, start_date, end_date):
    # Call the util function with the selected value
    value = util.resignedInPeriod(selected_value, start_date, end_date)
    return value

@callback(
    Output('service-in-period', 'children'),
    Input('dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_service_in_period(selected_value, start_date, end_date):
    # Call the util function with the selected value
    value = util.averageService(selected_value, start_date, end_date)
    return value

@callback(
    Output('turnover-rate', 'children'),
    Input('dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date')
)
def update_turnover_rate_in_period(selected_value, start_date, end_date):
    # Call the util function with the selected value
    value = util.turnoverRate(selected_value, start_date, end_date)
    return value