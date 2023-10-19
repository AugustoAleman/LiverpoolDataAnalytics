import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

def footer():

    footer_section = html.Div([
                    html.Div([
                        html.Div([
                            html.H3('Liverpool Human Analytics', className='footer-name'),
                            html.A('Descubre las historias detrás de los datos, explorando la esencia de la experiencia humana. La analítica se encuentra con la emoción, y las decisiones se basan en la ciencia y la humanidad. Juntos, desentrañamos el misterio de lo que nos hace únicos.', className='footer-description'),
                        ], className='footer-description-container'),

                        html.Div([
                            
                            html.Div([
                                html.H5('Navegación', className='footer-link-description'),
                                html.H5('Inicio', className='footer-link-element'),
                                html.H5('Sobre Nosotros', className='footer-link-element'),
                                html.H5('Servicios', className='footer-link-element'),
                                html.H5('Atributos', className='footer-link-element')
                            ], className='footer-link'),

                            html.Div([
                                html.H5('Información', className='footer-link-description'),
                                html.H5('Aviso legal', className='footer-link-element'),
                                html.H5('Política de privacidad', className='footer-link-element'),
                                html.H5('Centro de ayuda', className='footer-link-element'),
                                html.H5('Preguntas frecuentes', className='footer-link-element')
                            ], className='footer-link'),

                            
                            html.Div([
                                html.H5('Soporte', className='footer-link-description'),
                                html.H5('+52 (55)-55-55-55', className='footer-link-element'),
                                html.H5('analytics@liverpool.mx', className='footer-link-element'),
                                html.H5('Mario Pani 200, Lomas de Santa Fe, CDMX.', className='footer-link-element')
                            ], className='footer-link'),

                        ], className='footer-links')
                    ], className='footer-superior'),

                    html.Div([
                        html.Div([html.A('Copyright © 2023 - Todos los Derechos Reservados | Hecho por Augusto Alemán')], className = 'footer-copyright'),
                        html.Div([
                            html.Div([
                                html.Img(src='assets/src/1.png', className='footer-logo-image'),
                                html.Img(src='assets/src/2.png', className='footer-logo-image'),
                                html.Img(src='assets/src/3.png', className='footer-logo-image'),
                                html.Img(src='assets/src/4.png', className='footer-logo-image'),
                            ], className = 'footer-social-media-logos-container')
                        ], className = 'footer-social-media'),

                    ], className='footer-inferior')
                    ], className='footer')
    
    return footer_section