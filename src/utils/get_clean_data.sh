#!/bin/sh
DATA_PATH=/vol/bitbucket/hav16/imagenet
FILE_PATH=${DATA_PATH}/imagenet_clean.zip
if [ ! -d "$DATA_PATH" ]
then
    mkdir -p $DATA_PATH
    chmod 700 $DATA_PATH
fi


if [ ! -f "$FILE_PATH" ]
then
    wget -O ${FILE_PATH} https://www.dropbox.com/s/ijtoc7ra4gwqc7u/imagenet_clean.zip?dl=0
fi

unzip $FILE_PATH -d $DATA_PATH
