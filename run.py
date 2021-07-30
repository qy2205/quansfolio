import argparse
from load_database import reload_database
from flaskblog import app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-r', '--reset', action='store_true')
    parser.add_argument('-p', '--port', default=5000, type=int)
    parser.add_argument('--host', default='localhost')
    args = parser.parse_args()
    
    # reset db before running the application
    if args.reset:
        reload_database()
        
    # app.run(debug=args.debug, port=args.port, host=args.host)
    app.run(debug=True, port=args.port, host=args.host)
