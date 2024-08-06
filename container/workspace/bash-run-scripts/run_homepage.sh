HOMEPAGE_PORT=$1

#cd /workspace/python-scripts/landing-page

cd /workspace
mkdir landing-page
cd landing-page
git clone https://github.com/mlutils/beamds.git
cd beamds
git checkout dev
cd container/workspace/python-scripts/landing-page

python app.py --port $HOMEPAGE_PORT &
