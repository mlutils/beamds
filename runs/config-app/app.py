from flask import Flask, render_template, request, redirect, url_for
import sys
sys.path.insert(0, '../..')
from beam import ServeClusterConfig, resource
import argparse
import yaml
import os


app = Flask(__name__)

# Sample system parameters
system_params = {
    "cpu_cores": 4,
    "memory": "16GB",
    "disk_space": "500GB"
}

def read_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def write_config(file_path, config):
    with open(file_path, 'w') as f:
        yaml.dump(config, f)

def handle_submission(config_params, system_params):
    # Perform your logic here with the selected config and system parameters
    print("Received config params:", config_params)
    print("System params:", system_params)
    # Add your processing logic here, such as saving to a file, applying configuration, etc.

@app.route('/', methods=['GET', 'POST'])
def home():
    config_path = os.path.join('examples', 'orchestration_beamdemo.yaml')
    config_data = read_config(config_path)
    config_class = ServeClusterConfig(config_data)  # Assuming K8SConfig is already defined

    # Prepare parameters to pass to the template
    config_params = []
    for param in config_class.parameters:
        config_params.append({
            'name': param.name,
            'value': config_data.get(param.name),
            'type': param.type.__name__,
            'help': param.help or '',
            'icon': url_for('static', filename='images/param_icon.png')  # Placeholder for actual icons
        })

    if request.method == 'POST':
        selected_config = {}
        for param in config_class.parameters:
            value = request.form.get(param.name)
            if param.type == int:
                selected_config[param.name] = int(value)
            elif param.type == float:
                selected_config[param.name] = float(value)
            elif param.type == bool:
                selected_config[param.name] = value == 'on'
            elif param.type in [list, dict]:
                selected_config[param.name] = yaml.safe_load(value)
            else:
                selected_config[param.name] = value

        # Call the function with the selected config and system params
        handle_submission(selected_config, system_params)

        # Save the updated configuration
        write_config(config_path, selected_config)
        return redirect(url_for('home'))  # Refresh the page after submission

    return render_template('index.html', config_params=config_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=22001)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--manager', type=str, default=None)
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
