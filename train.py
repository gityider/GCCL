from comet_ml import Experiment, Optimizer

import argparse
import torch
import os
import gccl

log = gccl.utils.get_logger()


def main(args):
    gccl.utils.set_seed(args.seed)
    
    args.data = os.path.join(
        args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
    )

    # load data
    log.debug("Loading data from '%s'." % args.data)

    data = gccl.utils.load_pkl(args.data)
    log.info("Loaded data.")

    trainset = gccl.Dataset(data["train"], args)
    devset = gccl.Dataset(data["dev"], args)
    testset = gccl.Dataset(data["test"], args)

    log.debug("Building model...")
    
    model_file = "./model_checkpoints/model.pt"
    model = gccl.GCCL(args).to(args.device)
    opt = gccl.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = gccl.Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # Train
    log.info("Start training...")
    ret = coach.train()

    # Save.
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--dataset", type=str, default="iemocap", help="Dataset name.")
    ### adding other pre-trained text models
    parser.add_argument("--transformers", action="store_true", default=False)

    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )

    # Training parameters
    parser.add_argument(
        "--from_begin", action="store_false", help="Training from begin.", default=True
    )
    parser.add_argument("--model_ckpt", type=str, help="Training from a checkpoint.")

    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default="adam", 
                        choices=["sgd", "rmsprop", "adam", "adamw"], help="Name of optimizer.")
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")

    # Model parameters
    parser.add_argument(
        "--wp",
        type=int,
        default=5,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=5,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    parser.add_argument("--n_speakers", type=int, default=2, help="Number of speakers.")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden size of two layer GCN."
    )
    parser.add_argument( "--rnn", type=str, default="transformer", choices=["lstm", "gru", "transformer"], help="Type of RNN cell.")
    parser.add_argument("--class_weight", action="store_true", default=False, help="Use class weights in nll loss.")


    parser.add_argument(
        "--modalities", type=str, default="atv", help="Modalities",
    )
    parser.add_argument("--concat_gin_gout", action="store_true", default=False)
    parser.add_argument("--seqcontext_nlayer", type=int, default=2)
    parser.add_argument("--gnn_nheads", type=int, default=1)
    parser.add_argument("--num_bases", type=int, default=7)
    parser.add_argument("--use_highway", action="store_true", default=False)

    parser.add_argument("--seed", type=int, default=24, help="Random seed.")
    parser.add_argument("--tag", type=str, default="hyperparameters_opt")
    parser.add_argument('--M', default=100.0, help='Margin')

    args = parser.parse_args()

    args.dataset_embedding_dims = {
        "iemocap": {
            "a": 100,
            "t": 768,
            "v": 512,
            "at": 100 + 768,
            "tv": 768 + 512,
            "av": 612,
            "atv": 100 + 768 + 512,
        }
    }

    log.debug(args)

    main(args)
