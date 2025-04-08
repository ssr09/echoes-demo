import http.server
import socketserver
import webbrowser
import os

# Define port
PORT = 8000

# Change to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Set up a simple HTTP server
Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '': 'application/octet-stream',
}

# Create the server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving Echoes app at http://localhost:{PORT}")
    print("Open app.html in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Open browser automatically
    webbrowser.open(f'http://localhost:{PORT}/app.html')
    
    # Keep the server running
    httpd.serve_forever() 