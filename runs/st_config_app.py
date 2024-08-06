import json
import tempfile

import streamlit as st
import yaml
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '..')
from beam import K8SConfig, this_dir, resource
from beam import logger
from beam.path.utils import temp_local_file

# Path to the local image in the static directory
background_image_path = "static/st-beam.png"

# Ensure the image path is correct and the file exists
print(f"Background image path: {background_image_path}")

# Custom CSS for styling
st.markdown(f"""
    <style>
    html, body {{
        background: url('/{background_image_path}') no-repeat center center fixed;
        background-size: cover;
        margin: 0;
        padding: 0;
        height: 100vh;
    }}
    .gradient-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,0,150,0.3), rgba(0,204,255,0.3));
        z-index: -1;
    }}
    .stApp {{
        background-color: transparent;
    }}
    .param-name {{
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        padding: 5px 0;
        margin-bottom: 10px;
        position: relative;
        color: #000000; /* Ensure text is visible */
    }}
    .param-input {{
        margin-top: 10px;
    }}
    .param-name:hover::after {{
        content: attr(data-help);
        position: absolute;
        background: rgba(0,0,0,0.7);
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 14px;
        color: #fff;
        white-space: nowrap;
        z-index: 1000;
        left: 0;
        bottom: -25px;
    }}
    .terminal {{
        background: #000;
        color: #0f0;
        padding: 20px;
        border-radius: 5px;
        overflow-y: auto;
        height: 300px;
        margin-top: 20px;
    }}
    .tile-container {{
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
    }}
    .tile {{
        width: 45%;
        margin: 10px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        position: relative;
        color: #000000; /* Ensure text is visible */
    }}
    .tile input, .tile textarea, .tile select {{
        width: 100%;
    }}
    .launch-button {{
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        font-size: 20px;
        padding: 15px 30px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        text-align: center;
        margin-top: 20px;
    }}
    .launch-button:hover {{
        background-color: #155a8a;
    }}
    </style>
    <div class="gradient-overlay"></div>
""", unsafe_allow_html=True)

# Function to read the configuration file
def read_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Function to write the configuration file
def write_config(file_path, config):
    with open(file_path, 'w') as f:
        yaml.dump(config, f)


# Initialize session state for configuration
if 'config' not in st.session_state:
    conf_path = this_dir().parent.joinpath('examples', 'orchestration_beamdemo.yaml')
    st.session_state.config = read_config(conf_path)


def generate_ui(config):
    config_class = config.__class__
    param_values = {}
    st.markdown('<div class="tile-container">', unsafe_allow_html=True)
    with st.form("config_form"):
        for param in config_class.parameters:
            v = config.get(param.name)
            st.markdown(f'<div class="tile"><div class="param-name" data-help="{param.help}">{param.name}</div>',
                        unsafe_allow_html=True)
            if param.type == int:
                param_values[param.name] = st.number_input(param.help or param.name, value=int(v), format="%d",
                                                           key=param.name)
            elif param.type == float:
                param_values[param.name] = st.number_input(param.help or param.name, value=float(v), format="%f",
                                                           key=param.name)
            elif param.type == str:
                param_values[param.name] = st.text_input(param.help or param.name, value=v, key=param.name)
            elif param.type == bool:
                param_values[param.name] = st.checkbox(param.help or param.name, value=v, key=param.name)
            elif param.type == list:
                param_values[param.name] = json.loads(
                    st.text_area(param.help or param.name, value=json.dumps(v), key=param.name))
            elif param.type == dict:
                param_values[param.name] = json.loads(
                    st.text_area(param.help or param.name, value=json.dumps(v), key=param.name))
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit")
        if submitted:
            return param_values
    return None


def main():
    st.title("Configuration UI")

    config = K8SConfig(st.session_state.config)

    param_values = generate_ui(config)

    if param_values:
        logger.info("Updating configuration...")

        # Update session state with new values
        for param, value in param_values.items():
            st.session_state.config[param] = value

        # Save updated config to file
        conf_path = this_dir().parent.joinpath('examples', 'orchestration_beamdemo.yaml')
        write_config(conf_path, st.session_state.config)

    if st.button("Launch"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            user_config_path = Path(tmp_dir).joinpath('config.yaml')
            user_config_path.write_text(yaml.dump(st.session_state.config))

            logger.info(f"Running the deployment script from {this_dir().parent}...")
            base_dir = str(this_dir().parent)

            # Create a placeholder for the terminal output
            terminal_output = st.empty()

            process = subprocess.Popen(
                ["python", "-m", "examples.orchestration_beamdemo", str(user_config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=base_dir
            )

            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    terminal_output.code(output.strip(), language='bash')
                time.sleep(0.1)

            # Capture any remaining output
            stderr_output = process.stderr.read()
            if stderr_output:
                terminal_output.code(stderr_output.strip(), language='bash')


if __name__ == '__main__':
    main()
