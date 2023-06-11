cur_dir=`pwd`
cd /home/pavel/projects/horses/soft/python/morphometry/datasets/2023/segmentation
tree -iFPf '*.tps' --prune | grep tps > filelist.txt
mv filelist.txt $cur_dir
cd $cur_dir


