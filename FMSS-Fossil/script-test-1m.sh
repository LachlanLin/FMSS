#!/bin/bash
# This script is used to test our FMSS on ML1M.

runjava(){
java -Xmx40960m Main -lambda 0.001 -gamma 0.1 -xi 1.0 -L 3 -d 20 -fnTrainData ./ML1M/train_5.csv -fnTestData ./ML1M/test_5.csv -rho $2 -c $3 -n 6040 -m 3416 -topK 5 -num_iterations 1710 > ./results/ML1M_$1_rho_$2_c_$3.txt
}
for copy in 'run1' 'run2' 'run3'
do
rho='0'
c='0'
runjava ${copy} ${rho} ${c}
c='3'
for rho in '0' '1' '2'
do
runjava ${copy} ${rho} ${c}
done
rho='3'
for c in '0' '1' '2'
do
runjava ${copy} ${rho} ${c}
done
c='3'
runjava ${copy} ${rho} ${c}
done
