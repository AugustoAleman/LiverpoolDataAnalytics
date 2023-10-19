import dash
import dash_core_components as dcc
import dash_html_components as html

def navbar():
   navbar_component =  html.Div([
    dcc.Link(html.Img(src='assets/src/liverpool-logo.png', className='logo'), href = '/'),  # Customize logo path

    html.Div([
        html.Div([
        html.Div([
            dcc.Link(f'{page["name"]}', href=page['relative_path'], className='nav-option')
        ], className='nav-option-container') for page in dash.page_registry.values()
    ], className='nav-option-container'),
        html.Div(dcc.Link(html.Button('Salir', className='rounded-button'), href='/', className='rounded-button-container'))
    ], className='nav-options')

    ], className='navbar')

   return navbar_component