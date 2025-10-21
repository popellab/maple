#!/bin/bash

# QSP-LLM-Workflows Environment Setup Script
# Uses uv for fast, reliable Python environment management

set -e  # Exit on error

echo "=========================================="
echo "QSP-LLM-Workflows Environment Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed."
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✓ uv found: $(uv --version)"
echo ""

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "✓ System Python: $PYTHON_VERSION"
echo ""

# Create virtual environment with uv
echo "Creating virtual environment with uv..."
if [ -d "venv" ]; then
    echo "⚠️  venv directory already exists. Removing and recreating..."
    rm -rf venv
fi

uv venv venv --python python3
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies with uv
echo "Installing dependencies with uv..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    echo "✓ Dependencies installed from requirements.txt"
else
    echo "⚠️  No requirements.txt found. Skipping dependency installation."
    echo "   You may need to install dependencies manually."
fi
echo ""

# Check for .env file
echo "Checking for .env file..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << 'EOF'
# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here
EOF
    echo "✓ Created .env template"
    echo "  ⚠️  Please edit .env and add your OpenAI API key"
else
    echo "✓ .env file exists"
fi
echo ""

# Check for qsp-metadata-storage sibling repository
echo "Checking for qsp-metadata-storage repository..."
if [ ! -d "../qsp-metadata-storage" ]; then
    echo "⚠️  qsp-metadata-storage not found as sibling directory"
    echo ""
    echo "Clone it with:"
    echo "  cd .."
    echo "  git clone https://github.com/popellab/qsp-metadata-storage.git"
    echo "  cd qsp-llm-workflows"
    echo ""
else
    echo "✓ qsp-metadata-storage found"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Add your OpenAI API key to .env file"
echo ""
echo "3. Clone qsp-metadata-storage if not already done:"
echo "   cd .. && git clone https://github.com/popellab/qsp-metadata-storage.git"
echo ""
echo "4. Read the documentation:"
echo "   - CLAUDE.md for comprehensive codebase reference"
echo "   - docs-manuscript/COLLABORATOR_ONBOARDING.md for project overview"
echo ""
echo "Happy coding!"
