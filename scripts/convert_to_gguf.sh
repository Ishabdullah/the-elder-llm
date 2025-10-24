#!/bin/bash
#
# GGUF Conversion Script for The Elder LLM
# Converts Hugging Face model to GGUF format for mobile deployment
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔄 GGUF Conversion Tool - The Elder LLM${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Configuration
MODEL_DIR="${1:-./the-elder-merged}"
OUTPUT_DIR="${2:-./releases}"
OUTPUT_NAME="The_Elder.gguf"
QUANT_TYPE="Q4_K_M"  # 4-bit quantization, medium quality

echo -e "\n${YELLOW}Configuration:${NC}"
echo "  Model Directory: $MODEL_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Output Name: $OUTPUT_NAME"
echo "  Quantization: $QUANT_TYPE"

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "\n${RED}❌ Error: Model directory not found: $MODEL_DIR${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if llama.cpp is available
if [ ! -d "llama.cpp" ]; then
    echo -e "\n${YELLOW}📦 Cloning llama.cpp...${NC}"
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make
    cd ..
fi

# Install gguf if needed
if ! python3 -c "import gguf" 2>/dev/null; then
    echo -e "\n${YELLOW}📦 Installing gguf package...${NC}"
    pip install gguf
fi

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 1: Converting to FP16 GGUF${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

python3 llama.cpp/convert.py "$MODEL_DIR" \
    --outtype f16 \
    --outfile "$OUTPUT_DIR/the-elder-f16.gguf"

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Step 2: Quantizing to $QUANT_TYPE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

./llama.cpp/quantize \
    "$OUTPUT_DIR/the-elder-f16.gguf" \
    "$OUTPUT_DIR/$OUTPUT_NAME" \
    "$QUANT_TYPE"

# Clean up intermediate file
rm "$OUTPUT_DIR/the-elder-f16.gguf"

# Get file size
GGUF_SIZE=$(du -h "$OUTPUT_DIR/$OUTPUT_NAME" | cut -f1)

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ CONVERSION COMPLETE!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n📦 Output File: $OUTPUT_DIR/$OUTPUT_NAME"
echo -e "📊 File Size: $GGUF_SIZE"
echo -e "\n${GREEN}Ready for mobile deployment!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
