#!/bin/sh
data_path=/data/hav16/imagenet
clean_data_path=/data/hav16/clean
clean_file_path=${clean_data_path}/imagenet_clean.zip

if [ ! -d "$data_path" ]
then
    mkdir -p $data_path
    chmod 700 $data_path
fi


if [ ! -f "$clean_file_path" ]
then
    wget -O ${clean_file_path} https://www.dropbox.com/s/39e7kbukhgltd9t/clean_imagenet.zip?dl=0
fi

unzip $clean_file_path -d $data_path
