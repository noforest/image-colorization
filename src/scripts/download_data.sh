#!/usr/bin/sh


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
DATA_DIR="$ROOT_DIR/data"
COLORED_TRAIN="$DATA_DIR/colored/train"
COLORED_VALID="$DATA_DIR/colored/valid"

mkdir -p $COLORED_TRAIN
mkdir -p $COLORED_VALID

# train
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip -O colored_train.zip
mv colored_train.zip $COLORED_TRAIN/
cd $COLORED_TRAIN
unzip colored_train.zip

mv DIV2K_train_LR_bicubic/X2/* .
rm -rf colored_train


# valid
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip -O colored_valid.zip
mv colored_valid.zip $COLORED_VALID/
cd $COLORED_VALID
unzip colored_valid.zip

mv DIV2K_valid_LR_bicubic/X2/* .
rm -rf colored_valid
