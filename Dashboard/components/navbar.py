import dash
import dash_core_components as dcc
import dash_html_components as html

def navbar():
   navbar_component =  html.Div([
                html.Img(src='assets/src/liverpool-logo.png', className='logo'),  # Customize logo path

                html.Div([
                   html.Div(
                      dcc.Link(f'{page["name"]}', href = page['relative_path'])
                   , className = 'nav-option') for page in dash.page_registry.values()
                ], className='nav-options')

            ], className='navbar')
   return navbar_component