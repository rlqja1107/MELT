gpu=2
data="Behance" # fsq, Behance
python main.py --inference true --model MELT_SASRec --batch_size 128 --dataset $data --gpu $gpu --pareto_rule 0.5