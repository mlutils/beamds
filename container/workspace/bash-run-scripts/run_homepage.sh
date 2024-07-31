HOMEPAGE_PORT=$1

cd /workspace/python-scripts/landing-page
python app.py --port $HOMEPAGE_PORT &
