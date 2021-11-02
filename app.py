from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/playground')
def playground():
    return render_template('playground.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)