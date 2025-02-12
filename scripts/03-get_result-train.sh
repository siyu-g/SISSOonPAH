#!/bin/bash
# This script grep RMSE results from training data predictions

# Loops through each folder
for fcomplexity in 1 3 7
do
	
	for desc_dim in 1 2 3 4
	do
	folder=fcomp${fcomplexity}_dim${desc_dim}
	cd $folder
		rm train_rmse.txt
		# Grep RMSE from file in each folder
		for subfolder in $(ls -d cross_validate* | sort -V)
		do
		rmse=$(awk '/Prediction RMSE/ {split($0, a, ":"); split(a[2], b); print b[1]}' $subfolder/predict_Y.out)
		echo $subfolder $rmse >> train_rmse.txt
		done
	cd ../	
	done

done

