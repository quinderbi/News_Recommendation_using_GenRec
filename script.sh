echo "Running script.sh"

for model in FairGANModel DiffModel 
do
    for fold in 1 
    do
        echo "Training $model fold $fold"
        python train.py --model $model --fold $fold
    done
done