import datetime
from logging.config import dictConfig
from flask import Flask, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flaskext.markdown import Markdown
from sqlalchemy import event
from flask_simplemde import SimpleMDE


def create_app(test_config=None):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'  # change and create your own key
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Markdown editor
    app.config['SIMPLEMDE_JS_IIFE'] = True
    app.config['SIMPLEMDE_USE_CDN'] = True

    # see more at: https://flask.palletsprojects.com/en/1.1.x/config/#SEND_FILE_MAX_AGE_DEFAULT
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # option to be used during the development phase, prevents caching

    app.config['SQLALCHEMY_ECHO'] = False  # option for debugging -- should be set to False for production
    return app


# create app
app = create_app()

# database
db = SQLAlchemy(app)

# login manager
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# others
bcrypt = Bcrypt(app)
Markdown(app)
SimpleMDE(app)


# this function makes the web service api to be accessible only for the requests with valid tokens
@app.before_request
def before():
    if request.path.startswith('/api') \
            and request.path != '/api/token/public':
        # if the request goes to the API, and is different from the one to get a token, token should match
        if request.is_json and 'Authorization' in request.headers: # only JSON requests are allowed
            from flaskblog.models import Token
            token_string = request.headers['Authorization'].split(' ')[1]
            token = Token.query.filter_by(token=token_string).all()
            now = datetime.datetime.now()
            if len(token) == 0:
                abort(403)
            elif token[0].date_expired < now:
                abort(403)
        else:
            return abort(403)

    # limiting the addresses to only local and Chalmers network
    if not request.remote_addr.startswith('127.0.0') and not request.remote_addr.startswith('129.16.'):
        print('DENIED:', request.remote_addr, request.headers)
        abort(403)  # forbidden


# option to be used during the development phase, prevents caching
# comment when using it in production
# for more info check: https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.after_request
# and https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


from flaskblog import routes
