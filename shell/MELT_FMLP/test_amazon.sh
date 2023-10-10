gpu=2
data="Music" # Clothing, Sports, Beauty, Grocery, Automotive, Music
python main.py --inference true --model MELT_FMLP --batch_size 128 --dataset $data --gpu $gpu --pareto_rule 0.8