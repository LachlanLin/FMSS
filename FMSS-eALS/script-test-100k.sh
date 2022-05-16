#!/bin/bash
# This script is used to test our FMSS on ML100K

runjava(){
    java -Xmx40960m Main -lambda 0.01 -c_0 200 -d 20 -fnTrainingData ./ML100K/ML100K-$1-train -fnTestData ./ML100K/ML100K-$1-test -rho $2 -c $3 -n 943 -m 1682 -num_iterations 90 -topK 5 > ./results/ML100K_$1_rho_$2_c_$3.txt
}

for copy in 'copy1' 'copy2' 'copy3'
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
