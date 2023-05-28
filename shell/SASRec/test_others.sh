gpu=2 # GPU num
data="Behance" # fsq, Behance
python main.py --inference true --model SASRec --batch_size 128 --dataset $data --gpu $gpu --pareto_rule 0.5