gpu=2 # GPU num
data="Behance" # fsq, Behance
python main.py --inference true --model FMLP --batch_size 256 --dataset $data --gpu $gpu --pareto_rule 0.5