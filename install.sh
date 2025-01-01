#!/bin/bash

curl -L -o ~/Downloads/coin-image-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/mexwell/coin-image-dataset

curl -L -o ~/Downloads/coin-images.zip\
  https://www.kaggle.com/api/v1/datasets/download/wanderdust/coin-images

unzip ~/Downloads/coin-image-dataset.zip -d ~/Downloads/coin-image-dataset
unzip ~/Downloads/coin-images.zip -d ~/Downloads/coin-images

mv ~/Downloads/coin-image-dataset roman_coins
mv ~/Downloads/coin-images world_coins

mkdir -p dataset

cp -r world_coins/coins/data/train dataset/train
cp -r world_coins/coins/data/test dataset/test

rm -rf world_coins

for i in {212..271}; do
    mkdir -p dataset/train/$i
done

counter=212
find roman_coins -type f -name "*.png" | sort | while read -r file; do
    if [ $counter -le 271 ]; then
        cp "$file" "dataset/train/$counter/"
        files_in_dir=$(ls -1 "dataset/train/$counter" | wc -l)
        if [ $files_in_dir -eq 3 ]; then
            counter=$((counter + 1))
        fi
    fi
done

rm -rf roman_coins
rm -rf 