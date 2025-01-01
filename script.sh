echo "Running script.sh"

for model in FairGANModel DiffModel 
do
    for fold in 1 2 3 4 5
    do
        echo "Training $model fold $fold"
        python train.py --model $model --fold $fold
    done
done
