#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --all              Install all dependencies"
    echo "  --ds              Install data science dependencies"
    echo "  --llm             Install LLM dependencies"
    echo "  --serve           Install serving dependencies"
    echo "  --orchestration   Install orchestration dependencies"
    echo "  --dev             Install development dependencies"
    echo "  --help            Show this help message"
}

# Parse command line arguments
INSTALL_ALL=false
INSTALL_DS=false
INSTALL_LLM=false
INSTALL_SERVE=false
INSTALL_ORCHESTRATION=false
INSTALL_DEV=false

for arg in "$@"; do
    case $arg in
        --all)
            INSTALL_ALL=true
            ;;
        --ds)
            INSTALL_DS=true
            ;;
        --llm)
            INSTALL_LLM=true
            ;;
        --serve)
            INSTALL_SERVE=true
            ;;
        --orchestration)
            INSTALL_ORCHESTRATION=true
            ;;
        --dev)
            INSTALL_DEV=true
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            print_usage
            exit 1
            ;;
    esac
done

# Check if Python 3.12 is installed
if ! pyenv versions | grep -q "3.12"; then
    echo "Installing Python 3.12..."
    pyenv install 3.12.1
fi

# Set local Python version
pyenv local 3.12.1

# Install poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Configure poetry to use the correct Python version
poetry env use 3.12.1

# Install dependencies based on options
if [ "$INSTALL_ALL" = true ]; then
    echo "Installing all dependencies..."
    poetry install
else
    # Always install main dependencies
    echo "Installing main dependencies..."
    poetry install --no-root --only main

    # Install optional dependencies based on flags
    if [ "$INSTALL_DS" = true ]; then
        echo "Installing data science dependencies..."
        poetry install --no-root --only ds
    fi

    if [ "$INSTALL_LLM" = true ]; then
        echo "Installing LLM dependencies..."
        poetry install --no-root --only llm
    fi

    if [ "$INSTALL_SERVE" = true ]; then
        echo "Installing serving dependencies..."
        poetry install --no-root --only serve
    fi

    if [ "$INSTALL_ORCHESTRATION" = true ]; then
        echo "Installing orchestration dependencies..."
        poetry install --no-root --only orchestration
    fi

    if [ "$INSTALL_DEV" = true ]; then
        echo "Installing development dependencies..."
        poetry install --no-root --only dev
    fi
fi

# Activate the environment
echo "Activating virtual environment..."
source $(poetry env info --path)/bin/activate

echo "Environment setup complete!"
echo "To activate the environment in the future, run: source \$(poetry env info --path)/bin/activate" 