import subprocess
import sys
import re
import logging
from packaging import version as pver


# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def install_package(requirement):
    """
    Tries to install a package using pip and returns True if successful,
    False otherwise.
    """
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
        logging.info(f"Successfully installed {requirement}")
        return False
    except subprocess.CalledProcessError:
        logging.warning(f"Failed to install {requirement}")
        return True


def modify_requirement_version(requirement, new_version):
    """
    Modifies the version of a requirement string.
    """
    package_name, _, _ = parse_requirement(requirement)
    new_requirement = f"{package_name}~={new_version}"
    logging.info(f"Modified requirement from {requirement} to {new_requirement}")
    return new_requirement


def parse_requirement(requirement):
    """
    Parses a requirement string into package, version specifier, and version.
    """
    match = re.match(r"(\S+)([<>=~!]+)([\d\.]+)", requirement)
    if match:
        package_name, version_specifier, version = match.groups()
        return package_name, version_specifier, version
    else:
        return requirement, '', ''


def main(requirements_path):
    # Read the requirements file
    with open(requirements_path) as f:
        requirements = f.readlines()

    # Install each package
    for r in requirements:
        error = install_package(r)
        if error:

            package_name, version_specifier, version = parse_requirement(r)
            if version_specifier in ['', '==', '!=', '<=', '<']:
                logging.error(f"Failed to install {r}, skipping to the next package")

            else:

                # If the installation failed, try to install the latest version
                latest_version = pver.parse(version)
                new_version = f"{latest_version.major}.0"
                new_requirement = f"{package_name}~={new_version}"
                logging.info(f"Modified requirement from {r} to {new_requirement}")
                error = install_package(new_requirement)
                if error:
                    logging.error(f"Failed to install {new_requirement}, Skipping to the next package")
                    continue


if __name__ == '__main__':
    logging.info(f"Running gracefull-pip.py with file: {sys.argv[1]}")
    main(sys.argv[1])
