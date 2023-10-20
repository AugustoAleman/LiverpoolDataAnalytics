import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def login():
    login_component = html.Div([

        html.Div([
            html.Img(src = 'assets/src/liverpool-outside.png', className='login-image-banner-image')
        ], className= 'login-image-banner'),

        html.Div([
           html.Div([
                html.Div([html.Img(src = 'assets/src/liverpool-logo.png', className='login-logo-image')], className='login-logo'),
                html.Div([dcc.Input(id='username-input', type='text', placeholder='  Usuario', className='login-textfield')], className= 'login-text-field'),
                html.Div([dcc.Input(id='password-input', type='password', placeholder='  Contraseña', className='login-textfield')], className= 'login-text-field'),
                html.Div([html.Button('Ingresar', id='login-button', className='login-submit-button')])
           ], className='login-box-options')
        ], className= 'login-box-container')

    ], className='login-page')
    return login_component

'''
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import json

# Load your JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Create a list of users from the JSON data
users = data['Trabajadores']

app = dash.Dash(_name_)

app.layout = html.Div([
    html.H1("Liverpool Dashboard"),
    
    # Login Form
    html.Div([
        dcc.Input(id='username-input', type='text', placeholder='Username'),
        dcc.Input(id='password-input', type='password', placeholder='Password'),
        html.Button('Login', id='login-button'),
    ]),
    
    # Content to display after login
    html.Div(id='login-status'),
    dcc.Graph(id='my-graph'),
], style={'text-align': 'center'})

@app.callback(
    Output('login-status', 'children'),
    Output('my-graph', 'style'),
    Input('login-button', 'n_clicks'),
    Input('username-input', 'value'),
    Input('password-input', 'value')
)
def check_login(n_clicks, username, password):
    if n_clicks is None:
        return '', {'display': 'none'}
    
    # Check if the provided username and password match any user in the JSON data
    for user in users:
        if user['ID empleado'] == username and user['Contraseña'] == password:
            return 'Login successful!', {'display': 'block'}
    
    return 'Invalid username or password', {'display': 'none'}

if _name_ == '_main_':
    app.run_server(debug=True)
'''