algo=$1
gen=$2
# All args below are exponents for powers of 2
min_vals=$3
max_vals=$4
min_threads=$5
max_threads=$6

for i in $(seq $min_vals 2 $max_vals)
do
    for j in $(seq $min_threads $max_threads)
    do
        vals=$((2**$i))
        threads=$((2**$j))
        sbatch project.grace_job $vals $threads $gen $algo
    done
done