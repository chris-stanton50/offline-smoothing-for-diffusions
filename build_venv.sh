# Build the virtual environment for the project
mkdir venv
python3.11 -m venv ./venv/diffusions
source ./venv/diffusions/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Add the project itself to the virtual environment using a .pth file
# (Assumes that the cwd does not have spaces in the folder structure)
pwd > ./venv/diffusions/lib/python3.11/site-packages/sdes.pth
