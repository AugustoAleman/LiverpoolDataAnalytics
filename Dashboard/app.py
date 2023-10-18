import dash
import dash_html_components as html
from components import footer, navbar


app = dash.Dash(__name__, title='Liverpool Human Analytics', use_pages = True)  # Set the title
app.title = 'Liverpool Human Analytics'  # Alternate way to set the title

# Define the app layout
app.layout = html.Div([
    html.Link(href="assets\styles.css", rel="stylesheet"),
    # Navbar with logo and options
    navbar.navbar(),
    
    # Page Body
    dash.page_container,

    footer.footer()
])

# Define callback function(s) here if needed

if __name__ == '__main__':
    app.run_server(debug=True)
