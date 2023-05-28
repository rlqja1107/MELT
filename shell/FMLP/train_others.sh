gpu=2 # GPU num
data="Behance" # fsq, Behance
python main.py --inference false --model FMLP --e_max 201 --batch_size 256 --dataset $data --gpu $gpu --pareto_rule 0.5