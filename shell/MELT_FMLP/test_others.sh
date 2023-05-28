gpu=2
data="Behance" # fsq, Behance
python main.py --inference true --model MELT_FMLP --batch_size 256 --dataset $data --gpu $gpu --pareto_rule 0.5