gpu=2 # GPU num
data="Behance" # fsq, Behance
python main.py --inference false --model SASRec --e_max 201 --batch_size 128 --dataset $data --gpu $gpu --pareto_rule 0.5