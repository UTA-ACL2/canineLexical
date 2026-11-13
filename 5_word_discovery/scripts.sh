# For training
python -m launch_train --input_directory=./short-dog-no-noise --output_directory=./ --grammar_file=./short-dog-no-noise/grammar.unigram --number_of_documents=31292 --batch_size=100

# For testing
nohup python -m launch_test --input_directory short-dog --model_directory short-dog/24Jun14-222049-D31294-P10-S10-B128-O1000-t512-k0.8-Ggrammar.unigram --non_terminal_symbol Word --number_of_samples 1 > short-dog/logs/test0615.log 2>&1 &