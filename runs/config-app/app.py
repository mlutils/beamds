from flask import Flask, render_template, request, redirect, url_for
import sys
sys.path.insert(0, '../..')
from beam import ServeClusterConfig, resource
from beam.orchestration.config import RnDClusterConfig
import argparse
import yaml
import os


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=22001)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--manager', type=str, default=None)
parser.add_argument('--config-file', type=str, default=None)
parser.add_argument('--application', type=str, default=None)
args = parser.parse_args()


app = Flask(__name__)

def read_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def write_config(file_path, config):
    with open(file_path, 'w') as f:
        yaml.dump(config, f)

def handle_submission(config_params):

    manager = resource(args.manager)

    if args.application == 'rnd':
        application = getattr(manager, 'launch_rnd_cluster')
    elif args.application == 'serve':
        application = getattr(manager, 'launch_serve_cluster')
    else:
        raise ValueError("Invalid application type")

    print("Received config params:")
    print(config_params)
    try:
        application(config_params)
    except Exception as e:
        print(f"Error launching application: {e}")


@app.route('/', methods=['GET', 'POST'])
def home():
    config_data = resource(args.config_file).read()

    if args.application == 'rnd':
        config_class = RnDClusterConfig(config_data, load_script_arguments=False)
    elif args.application == 'serve':
        config_class = ServeClusterConfig(config_data, load_script_arguments=False)
    else:
        raise ValueError("Invalid application type")

    # Prepare parameters to pass to the template
    config_params = []
    config_classes = config_class.__class__.mro()
    for cc in config_classes:
        if not hasattr(cc, 'parameters'):
            continue
        for param in cc.parameters:
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

        if args.application == 'rnd':
            config_class = RnDClusterConfig(config_data, load_script_arguments=False, **selected_config)
        elif args.application == 'serve':
            config_class = ServeClusterConfig(config_data, load_script_arguments=False, **selected_config)
        else:
            raise ValueError("Invalid application type")

        # Call the function with the selected config and system params
        handle_submission(config_class)

        # Save the updated configuration
        # write_config(config_path, selected_config)
        return redirect(url_for('home'))  # Refresh the page after submission

    if args.application == 'rnd':
        index_file = 'index_rnd.html'
    elif args.application == 'serve':
        index_file = 'index_serve.html'
    else:
        raise ValueError("Invalid application type")
    return render_template(index_file, config_params=config_params)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)