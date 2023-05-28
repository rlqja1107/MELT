gpu=2 # GPU num
data="Behance" # fsq, Behance

# ========Hyperparameter==========
lamb_u=0.1
lamb_i=0.4
e_max=180
# ================================
python main.py --inference false --dataset $data --gpu $gpu --model MELT_SASRec --lamb_u $lamb_u --lamb_i $lamb_i --e_max $e_max --pareto_rule 0.5 --batch_size 128 --lr 0.005
