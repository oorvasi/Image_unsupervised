from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def hello_world():
    """Print 'Image Classification!' as the response body."""
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)