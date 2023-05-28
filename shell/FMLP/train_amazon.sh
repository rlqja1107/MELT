gpu=2 # GPU num
data="Music" # Clothing, Sports, Beauty, Grocery, Automotive, Music
python main.py --inference false --model FMLP --e_max 201 --batch_size 256 --dataset $data --gpu $gpu --pareto_rule 0.8