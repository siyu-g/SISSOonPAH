#!/bin/bash
# This script creates folders for cross validation

# Create input for each folder
# Loops through different fcomplexity and descriptor dimensions
for fcomplexity in 1 3 7
do
	
	for desc_dim in 1 2 3 4
	do
	folder=fcomp${fcomplexity}_dim${desc_dim}
	mkdir $folder
	cd $folder
	cp -r ../data_splits/cross_validate* .
	
	# Change fcomplexity and descriptor dimention by current setting in the folder
	sed "s/desc_dim=1/desc_dim=$desc_dim/g" ../SISSO.in > ./SISSO.in
	sed -i "s/fcomplexity=1/fcomplexity=$fcomplexity/g" SISSO.in
		
		# Submit jobs in each folder
		for subfolder in `ls -d cross_validate*`
		do
		cd $subfolder
		cp ../../submit-train.sh .
		cp ../SISSO.in .
		sbatch submit-train.sh
		cd ../
		done
	cd ../	
	done

done

