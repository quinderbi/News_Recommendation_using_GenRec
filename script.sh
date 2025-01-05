echo "Running script.sh"

for model in FairGANModel DiffModel 
do
    for fold in 1 2 3 4 5
    do
        echo "Training $model fold $fold"
        python train.py --ds_name ebnerd_demo --model $model --fold $fold --worker_name "epoch 100 filter"
    done
done
