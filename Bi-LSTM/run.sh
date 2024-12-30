# # Wheat Yield
python run.py \
    --phenotype Yield \
    --training_scenario S1 \
    --batch_size 256 \
    --num_layers 2 
    wait

python run.py \
    --phenotype Yield \
    --training_scenario S2 \
    --batch_size 256 \
    --num_layers 2 
    wait

python run.py \
    --phenotype Yield \
    --training_scenario S3 \
    --batch_size 256 \
    --num_layers 6
    wait


# # Plant Height
python run.py \
    --phenotype Ph \
    --training_scenario S1 \
    --batch_size 128 \
    --num_layers 2 
    wait

python run.py \
    --phenotype Ph \
    --training_scenario S1 \
    --batch_size 128 \
    --num_layers 2
    wait

python run.py \
    --phenotype Ph \
    --training_scenario S1 \
    --batch_size 128 \
    --num_layers 2 
    wait
