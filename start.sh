# This script is used by Glitch to start your Python application with Gunicorn
gunicorn --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 app:app