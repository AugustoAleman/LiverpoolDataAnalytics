import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def navbar():
   navbar_component =  html.Div([
                html.Img(src='assets/src/liverpool-logo.png', className='logo'),  # Customize logo path
                html.Div([
                    html.Div(html.A('INICIO'), style = {'text-decoration': 'underline'}, className='nav-option'),
                    html.Div(html.A('USUARIOS'), className='nav-option'),
                    html.Div(html.A('PANEL DE CONTROL'), className='nav-option'),
                    html.Div(html.Button('SALIR', className='rounded-button')),
                ], className='nav-options')

            ], className='navbar')
   return navbar_component