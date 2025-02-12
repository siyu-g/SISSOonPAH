#!/bin/bash
# This script loops through all cross validation folders and submit SISSO training jobs

# Create input for each folder

# Loops through different fcomplexity and descriptor dimensions
for fcomplexity in 1 3 7
do
	
	for desc_dim in 1 2 3 4
	do
	folder=fcomp${fcomplexity}_dim${desc_dim}
	cd $folder
		
		# Submit jobs in each folder
		for subfolder in `ls -d cross_validate*`
		do
		cd $subfolder
		cp ../../submit-predict.sh .
		# Submit train RMSE prediction
		cp ../../data_splits/predict-train.dat ./predict.dat
		cat > SISSO_predict_para << EOF
91  ! Number of materials in the file predict.dat (same format with train.dat)
14  ! Number of features in the file predict.dat
$desc_dim  ! Highest dimension of the models to be read from SISSO.out
1  ! Property type 1:continuous or 2:categorical 
EOF
		sbatch submit-predict.sh
		cd ../
		done
	cd ../	
	done

done

