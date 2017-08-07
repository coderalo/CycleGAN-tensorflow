#Code is borrowed and modified from https://github.com/junyanz/CycleGAN.git


FILE=$1
DATA_DIR=$2

if ! [ -d $DATA_DIR ]; then
	echo "The data directory doesn't exist...create it"
	mkdir $DATA_DIR
fi

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
ZIP_FILE=$DATA_DIR/$FILE.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d $DATA_DIR
rm $ZIP_FILE
