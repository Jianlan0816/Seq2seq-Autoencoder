# NLP take-home: Derivatives

### Files 
- network.txt: parameters summarization
- requirements.txt: package version list

# ./seq2seq_derivative/data.py: partition train.txt into training and testing
# ./seq2seq_derivative/layers.py: layers construction
# ./seq2seq_derivative/main.py: test trained model on test.txt
# ./seq2seq_derivative/test.py: unit test
# ./seq2seq_derivative/train.py: model training
# ./seq2seq_derivative/utils.py: utils functions
# ./seq2seq_derivative/data: train.txt, splitted into training set and testing set

### Command
python train.py \
    "models/best" \
    --gpus 1 \
    --gradient_clip_val 1 \
    --max_epochs 10 \
    --val_check_interval 0.2
    
python main.py models/best --data_path [path/to/test.txt]
