import json
import streamlit as st
import yaml
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, '..')
from beam import K8SConfig, this_dir
from beam import logger

# Path to the local image
background_image_path = this_dir().joinpath("resources", "st-beam.png").as_posix()

# Custom CSS for styling
st.markdown(f"""
    <style>
    body {{
        background: url('{background_image_path}') no-repeat center center fixed;
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
    .param-name {{
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        padding: 5px 0;
    }}
    .param-input {{
        display: none;
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

def generate_ui(config):
    config_class = config.__class__
    param_values = {}
    st.markdown('<div class="tile-container">', unsafe_allow_html=True)
    for param in config_class.parameters:
        v = config.get(param.name)
        st.markdown(f'<div class="tile"><div class="param-name hover-help" data-help="{param.help}" onclick="toggleParamInput(\'{param.name}\')">{param.name}</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div id="input-{param.name}" class="param-input">', unsafe_allow_html=True)
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
        st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return param_values

def save_to_yaml(param_values, filepath='config.yaml'):
    with open(filepath, 'w') as yaml_file:
        yaml.dump(param_values, yaml_file, default_flow_style=False)

def main():
    st.title("Configuration UI")

    conf_path = this_dir().parent.joinpath('examples', 'orchestration_beamdeploy.yaml')
    config = K8SConfig(conf_path)

    param_values = generate_ui(config)

    # Custom styled button using HTML and CSS
    if st.button("Launch"):
        logger.info("Launching the deployment...")
        save_to_yaml(param_values)

        script_path = str(this_dir().parent.joinpath('examples', 'orchestration_beamdeploy.py'))

        # Create a placeholder for the terminal output
        terminal_output = st.empty()

        process = subprocess.Popen(
            ["python", script_path, "config.yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
