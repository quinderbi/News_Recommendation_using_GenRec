echo "Running script.sh"

# param input == 1
if [ "$#" -eq 1 ] 
then
    workers=$1
else
    workers= "default_worker"
fi

for model in FairGANModel DiffModel 
do
    for fold in 1 
    do
        echo "Training $model fold $fold"
        python train.py --model $model --fold $fold --workers $workers
    done
done