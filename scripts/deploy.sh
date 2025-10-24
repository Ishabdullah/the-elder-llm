#!/bin/bash
#
# The Elder LLM - Deployment Script
# Automates the complete training and deployment process
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check environment variables
check_env() {
    print_header "🔍 Checking Environment"

    if [ -z "$HF_TOKEN" ]; then
        print_error "HF_TOKEN not set"
        echo "Set it with: export HF_TOKEN='your_token'"
        exit 1
    else
        print_success "HF_TOKEN is set"
    fi

    if [ -z "$GH_TOKEN" ]; then
        print_warning "GH_TOKEN not set (optional)"
    else
        print_success "GH_TOKEN is set"
    fi

    echo ""
}

# Deploy to GitHub
deploy_github() {
    print_header "📤 Deploying to GitHub"

    if [ ! -d ".git" ]; then
        print_info "Initializing git repository..."
        git init
        git branch -m main
    fi

    print_info "Adding files..."
    git add .

    print_info "Creating commit..."
    git commit -m "Deploy The Elder LLM - Complete training pipeline" || true

    if [ -n "$GH_TOKEN" ]; then
        print_info "Setting up remote..."
        git remote add origin "https://$GH_TOKEN@github.com/Ishabdullah/the-elder-llm.git" 2>/dev/null || \
        git remote set-url origin "https://$GH_TOKEN@github.com/Ishabdullah/the-elder-llm.git"

        print_info "Pushing to GitHub..."
        git push -u origin main --force

        print_success "Deployed to GitHub: https://github.com/Ishabdullah/the-elder-llm"
    else
        print_warning "Cannot push without GH_TOKEN"
        echo "Push manually with: git push -u origin main"
    fi

    echo ""
}

# Open Colab notebook
open_colab() {
    print_header "🚀 Training Instructions"

    local colab_url="https://colab.research.google.com/github/Ishabdullah/the-elder-llm/blob/main/notebooks/The_Elder_Training.ipynb"

    echo ""
    print_info "To train The Elder model:"
    echo ""
    echo "  1. Open this URL in your browser:"
    echo "     $colab_url"
    echo ""
    echo "  2. Enable GPU:"
    echo "     Runtime → Change runtime type → T4 GPU"
    echo ""
    echo "  3. Add Secrets (🔑 icon on left):"
    echo "     - HF_TOKEN: Your Hugging Face token"
    echo "     - GH_TOKEN: Your GitHub token"
    echo ""
    echo "  4. Run All Cells:"
    echo "     Runtime → Run all"
    echo ""
    echo "  5. Wait ~1 hour for training to complete"
    echo ""

    # Try to open in browser if termux-open-url available
    if command -v termux-open-url &> /dev/null; then
        print_info "Opening Colab in browser..."
        termux-open-url "$colab_url"
    fi

    echo ""
}

# Show final summary
show_summary() {
    print_header "📊 Deployment Summary"

    echo ""
    print_info "Repository Structure:"
    echo "  ├── data/the_elder_dataset.jsonl (50+ examples)"
    echo "  ├── configs/the_elder_system_prompt.txt"
    echo "  ├── notebooks/The_Elder_Training.ipynb"
    echo "  └── scripts/convert_to_gguf.sh"
    echo ""

    print_info "Next Steps:"
    echo "  1. Open the Colab notebook (link above)"
    echo "  2. Run all cells to train The Elder"
    echo "  3. Download The_Elder.gguf from Hugging Face"
    echo "  4. Deploy to your Android device"
    echo ""

    print_info "Key Links:"
    echo "  • GitHub: https://github.com/Ishabdullah/the-elder-llm"
    echo "  • Hugging Face: https://huggingface.co/Ishabdullah/The_Elder"
    echo "  • Colab: https://colab.research.google.com/github/Ishabdullah/the-elder-llm/blob/main/notebooks/The_Elder_Training.ipynb"
    echo ""

    print_success "Deployment complete! Ready to train."
    echo ""
}

# Main execution
main() {
    clear
    print_header "🧙 The Elder LLM - Deployment Script"
    echo ""

    # Check environment
    check_env

    # Deploy to GitHub
    deploy_github

    # Show training instructions
    open_colab

    # Show summary
    show_summary

    print_header "✨ May The Elder guide you on your path ✨"
}

# Run
main
