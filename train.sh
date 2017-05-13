DEG=2
PRE="pca"
PARAM=""
SAVE="../tmp/model"

for n_tree in $1
do
    for min_num in $2
    do
        for frac in $3
        do

            python main.py --save $SAVE --task train --deg $DEG --pre $PRE --param_rf $n_tree,$min_num,$frac
            #acc=$(python main.py --save $SAVE --task train --deg $DEG --pre $PRE --param_rf $n_tree,$min_num,$frac)
            #echo $n_tree,$min_num,$frac,$acc
        done
    done
done
