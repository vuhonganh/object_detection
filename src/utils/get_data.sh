#!/bin/sh
# get raw data (user should not run this file)

DATA_PATH=/data/hav16/imagenet
FILE_PATH=${DATA_PATH}/imagenet.zip
if [ ! -d "$DATA_PATH" ]
then
    mkdir -p $DATA_PATH
    chmod 700 $DATA_PATH
fi


if [ ! -f "$FILE_PATH" ]
then
    wget -O ${FILE_PATH} https://www.dropbox.com/s/dz6kjgmlanju1pr/imagenet.zip?dl=0
fi

# old file is .tar.gz
# tar -xzf $FILE_PATH -C $DATA_PATH

# current file is compressed in zip format
unzip $FILE_PATH -d $DATA_PATH


cd $DATA_PATH

for TARZ_FILE in *.tar.gz 
do
 	tar -xzf $TARZ_FILE 
done

for TAR_FILE in *.tar 
do 
	tar -xf $TAR_FILE 
done

cd -
