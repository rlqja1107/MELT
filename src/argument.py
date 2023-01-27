import argparse

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SASRec', choices=["SASRec", "FMLP", "MELT_SASRec", "MELT_FMLP"])
    parser.add_argument('--dataset', default='Music', help="Datasets")
    parser.add_argument('--batch_size', default=128, type=int, help="FMLP:256, SASRec: 128")
    parser.add_argument('--maxlen', default=50, type=int, help="Constrain the max length")
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--hidden_units', default=50, type=int, help="d")
    parser.add_argument('--num_blocks', default=2, type=int, help="Layer")
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--l2_emb', default=0.0, type=float, help="Only for SASRec")
    parser.add_argument('--gpu', default="0", type=str, help="Assign memory to certain gpu")
    parser.add_argument('--inference', default=True, type=str2bool)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--n_negative_samples', default=100, type=int)
    
    parser.add_argument('--branch_batch_size', default=32, type=int, help="For FMLP")
    parser.add_argument('--e_max', default=41, type=int, help="Max epoch")
    parser.add_argument('--pareto_rule', default=0.8, type=float, help="Amazon: 0.8, Others: 0.5")
    parser.add_argument('--lamb_u', default=0.2, type=float, help="Regularizaton for user branch")
    parser.add_argument('--lamb_i', default=0.3, type=float, help="Regularizaton for item branch")
    args = parser.parse_args()
    return args