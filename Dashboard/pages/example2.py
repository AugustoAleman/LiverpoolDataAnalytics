import dash
from dash import html

dash.register_page(__name__)

layout = html.Div([
    html.H1('Esta es la página de ejemplo 2'),
    html.Div('Este es el contenido de la página de ejemplo 2 .'),
])