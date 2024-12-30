# Wheat Yield 
python run.py \
    --phenotype Yield \
    --training_scenario S1 \
    --batch_size 256 \
    --num_layers 2 \
    --mlp_ratio 16
    wait 

python run.py \
    --phenotype Yield \
    --training_scenario S2 \
    --batch_size 256 \
    --num_layers 2 \
    --mlp_ratio 16
    wait

python run.py \
    --phenotype Yield \
    --training_scenario S3 \
    --batch_size 256 \
    --num_layers 6 \
    --mlp_ratio 64
    wait


# Plant Height
python run.py \
    --phenotype Ph \
    --training_scenario S1 \
    --batch_size 256 \
    --num_layers 2 \
    --mlp_ratio 16
    wait

python run.py \
    --phenotype Ph \
    --training_scenario S2 \
    --batch_size 256 \
    --num_layers 2 \
    --mlp_ratio 16
    wait

python run.py \
    --phenotype Ph \
    --training_scenario S3 \
    --batch_size 256 \
    --num_layers 2 \
    --mlp_ratio 16 
    wait
