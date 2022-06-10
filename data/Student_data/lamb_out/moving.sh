#!/usr/bin/bash

for fold in ./lamb*/
do
	echo ${fold}
#	mv ${fold}conc_0.25_out1alloc_matrix.csv ${fold}conc_0.25_out1/alloc_matrix.csv
#	mv ${fold}conc_0.5_out1alloc_matrix.csv ${fold}conc_0.5_out1/alloc_matrix.csv
#	mv ${fold}conc_1_out1alloc_matrix.csv ${fold}conc_1_out1/alloc_matrix.csv
	rm -rf ${fold}conc_0.25_out0/
	rm -rf ${fold}conc_0.5_out0/
	rm -rf ${fold}conc_1_out0/
done
