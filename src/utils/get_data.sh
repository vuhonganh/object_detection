#!/bin/sh
# get raw data and clean it
# put -x next to #!/bin/sh to debug

#data_path=/data/hav16/imagenet
data_path=/data/hav16/test
data_path_raw=/data/hav16/raw
data_path_clean=/data/hav16/clean
file_path=${data_path_raw}/imagenet.zip
clean_file_path=${data_path_clean}/clean_imagenet.zip

mkdir -p $data_path
chmod 700 $data_path
mkdir -p $data_path_raw
chmod 700 $data_path_raw
mkdir -p $data_path_clean
chmod 700 $data_path_clean

if [ ! -f "$file_path" ]
then
    wget -O ${file_path} https://www.dropbox.com/s/dz6kjgmlanju1pr/imagenet.zip?dl=0
fi

# old file is .tar.gz
# tar -xzf $file_path -C $data_path

# current file is compressed in zip format
# -o to force overwrite
unzip -o $file_path -d $data_path


cd $data_path > /dev/null

for tarz_file in *.tar.gz
do
 	tar -xzf $tarz_file
 	rm $tarz_file # remove compressed file when done
done

for tar_file in *.tar
do
	tar -xf $tar_file
	rm $tar_file  # remove compressed file when done
done
cd - > /dev/null

python clean_data.py "$data_path"

cd $data_path > /dev/null
rm -rf Annotation/
zip -qr $clean_file_path .
cd - > /dev/null
