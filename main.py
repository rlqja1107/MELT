import torch
from src.argument import parse_args


def main():
    args = parse_args() 
    torch.set_num_threads(4)

    if args.model == "SASRec":
        from models.SASRec import Trainer
        embedder = Trainer(args)
        
    elif args.model == "FMLP":
        from models.FMLP import Trainer
        embedder = Trainer(args)
        
    elif args.model == "MELT_SASRec":
        from models.MELT_SASRec import Trainer
        embedder = Trainer(args)
    
    elif args.model == "MELT_FMLP":
        from models.MELT_FMLP import Trainer
        embedder = Trainer(args)
    
    if args.inference:
        embedder.test()
    else:
        embedder.train()

if __name__ == "__main__":
    main()