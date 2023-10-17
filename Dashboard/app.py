import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import datetime
from modules import util
from components import footer, navbar


app = dash.Dash(__name__, title='Liverpool Human Analytics')  # Set the title
app.title = 'Liverpool Human Analytics'  # Alternate way to set the title
app.favicon = '../assets/src/logo-pequenio.png'

# Define the app layout
app.layout = html.Div([
    html.Link(href="assets\styles.css", rel="stylesheet"),
    # Navbar with logo and options
    navbar.navbar(),
    # Two-column layout
    html.Div([
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
                id='date-picker',
                start_date=datetime(2019, 1, 1),
                end_date=datetime(2023, 1, 31),
            ),
            html.H4('Selecciona ubicaciones', className='picker-description'),
            html.Div([
                dcc.Dropdown(['Liverpool Polanco', 'Suburbia Coapa', 'CeDis Nacional'], 'Liverpool Polanco', multi=True)
            ]),
        ], className='left-column', style={'width': '30%'}),
        
        # Right column with charts
        html.Div([
            html.Div([
                html.Div([
                    html.Div('Tendencia de renuncias', className='chart-title'),
                    html.Hr(className='divider'),  # Horizontal line
                    dcc.Graph(
                        figure=util.resignationOverMonnths(),
                    ),
                ], className = 'chart-container'),
                html.Div([
                    html.Div('Renuncias en Sectores Estrátegicos', className='chart-title'),
                    html.Hr(className='divider'),
                    dcc.Graph(
                        figure=util.resignedPerArea(),
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
    ], className='two-column-layout'),
    footer.footer()
])

# Define callback function(s) here if needed

if __name__ == '__main__':
    app.run_server(debug=True)
