#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        exit 1
    fi
}

cleanup() {
    echo -e "${BLUE}Cleaning up temporary files...${NC}"
    # rm -rf ~/Downloads/coin-image-dataset.zip
    rm -rf ~/Downloads/coin-images.zip
    # rm -rf ~/Downloads/coin-image-dataset
    rm -rf ~/Downloads/coin-images
}

set -e
trap cleanup ERR

echo -e "${BLUE}Starting dataset download and preparation...${NC}"

# Create directories
echo "Creating directories..."
mkdir -p dataset
check_status "Directory creation"

# Download datasets
echo "Downloading datasets..."
# curl -L -o ~/Downloads/coin-image-dataset.zip \
#     https://www.kaggle.com/api/v1/datasets/download/mexwell/coin-image-dataset
# check_status "Downloaded dataset 1"

curl -L -o ~/Downloads/coin-images.zip \
    https://www.kaggle.com/api/v1/datasets/download/wanderdust/coin-images
check_status "Downloaded dataset 2"

# Extract
echo "Extracting archives..."
# unzip -q ~/Downloads/coin-image-dataset.zip -d ~/Downloads/coin-image-dataset
# check_status "Extracted dataset 1"

unzip -q ~/Downloads/coin-images.zip -d ~/Downloads/coin-images
check_status "Extracted dataset 2"

# Move directories
echo "Moving directories..."
# mv ~/Downloads/coin-image-dataset roman_coins
mv ~/Downloads/coin-images world_coins
check_status "Moved directories"

# Copy data
echo "Copying training data..."
cp -r world_coins/coins/data/train dataset
cp -r world_coins/coins/data/test dataset
check_status "Training data copy"

# Clean up world coins
rm -rf world_coins

# Create roman coins
# echo "Creating class folders..."
# for i in {212..271}; do
#     mkdir -p dataset/train/$i
# done
check_status "Class folder creation"

# Copy files in groups of 3
# echo "Copying dataset 1..."
# counter=212
# total_files=$(find roman_coins -type f -name "*.png" | wc -l)
# current_file=0

# find roman_coins -type f -name "*.png" | sort | while read -r file; do
#     if [ $counter -le 271 ]; then
#         cp "$file" "dataset/train/$counter/"
#         current_file=$((current_file + 1))
#         files_in_dir=$(ls -1 "dataset/train/$counter" | wc -l)
        
#         progress=$((current_file * 100 / total_files))
#         echo -ne "Progress: ${progress}% (Folder $counter) \r"
        
#         if [ $files_in_dir -eq 3 ]; then
#             counter=$((counter + 1))
#         fi
#     fi
# done
# echo -e "\n"
# check_status "Dataset 1 copy"

# Clean up roman_coins
echo "Performing final cleanup..."
# rm -rf roman_coins
cleanup
check_status "Cleanup"

echo -e "${GREEN}Dataset preparation completed successfully!${NC}"