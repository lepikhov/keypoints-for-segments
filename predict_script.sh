image=$1
python predict.py -s "head" -i $image
python predict.py -s "neck" -i $image
python predict.py -s "body" -i $image
python predict.py -s "front leg" -i $image 
python predict.py -s "rear leg" -i $image
python predict.py -s "horse" -i $image