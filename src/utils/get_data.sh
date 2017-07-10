#!/bin/sh
DATA_PATH=/vol/bitbucket/hav16/imagenet
FILE_PATH=${DATA_PATH}/imagenet.tar.gz
if [ ! -d "$DATA_PATH" ]
then
    mkdir -p $DATA_PATH
    chmod 700 $DATA_PATH
fi


if [ ! -f "$FILE_PATH" ]
then
    wget -O ${FILE_PATH} https://www.dropbox.com/s/mi0e17x5ubng1mn/imagenet.zip?dl=0
fi

tar -xzf $FILE_PATH -C $DATA_PATH

cd $DATA_PATH

for TARZ_FILE in bbox*.tar.gz 
do
 	tar -xzf $TARZ_FILE 
done

for TAR_FILE in *.tar 
do 
	tar -xf $TAR_FILE 
done

cd -