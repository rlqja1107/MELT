gpu=2 # GPU num
data="Music" # Clothing, Sports, Beauty, Grocery, Automotive, Music

# ========Hyperparameter==========
lamb_u=0.1
lamb_i=0.4
e_max=180
# ================================
python main.py --inference false --dataset $data --gpu $gpu --model MELT_FMLP --lamb_u $lamb_u --lamb_i $lamb_i --e_max $e_max --pareto_rule 0.8 --batch_size 256
