#!/bin/bash
# PolluSense Dashboard Startup Script

cd "$(dirname "$0")"

echo "🌍 Starting PolluSense Dashboard..."
echo ""
echo "📍 Location: $(pwd)"
echo "🐍 Python: .venv/bin/python"
echo "📊 Streamlit: .venv/bin/streamlit"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found!"
    echo "Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if SHAP is installed
.venv/bin/python -c "import shap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  SHAP not installed. Installing now..."
    .venv/bin/pip install shap -q
    echo "✅ SHAP installed!"
    echo ""
fi

echo "🚀 Launching Streamlit..."
echo ""
echo "📱 Dashboard will open at:"
echo "   • http://localhost:8501"
echo "   • http://$(ipconfig getifaddr en0 2>/dev/null || echo 'network-ip'):8501"
echo ""
echo "💡 Press Ctrl+C to stop"
echo ""
echo "─────────────────────────────────────────────────"
echo ""

# Start Streamlit
.venv/bin/streamlit run web/app.py
