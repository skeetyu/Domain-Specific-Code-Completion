import os
import argparse
import logging
import json
import torch

from tqdm import tqdm

from classifier import Classifier
from utils import set_seed

def logger_setup(log_dir, log_file):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def main(args):
    # Setup logger
    logger = logger_setup(args.log_dir, args.log_file)
    logger.info("Training/Evaluation parameters %s", args)

    set_seed(args.seed)
    
    if args.do_train:      
        classifier = Classifier(
            input_size=args.input_size,
            scale_size=args.scale_size,
            dropout=args.dropout,
            train_dir=args.train_dir,
            validation_dir=args.validation_dir,
            model_path=None,
            model_type=args.model_type,
            device=torch.device(f'cuda:{args.cuda}'),
            logger=logger
        )
        classifier.train_model(
            output_dir=args.output_dir,
            epoch=args.epoch,
            save_step=args.save_step,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_step=args.lr_step,
            lr_gamma=args.lr_gamma,
            threshold=args.threshold
        )

    if args.do_eval:
        if args.eval_all_ckpt:
            for i in range(29):
                model_path = args.ckpt
                model_path = model_path[:model_path.rfind('/')] + f'/ckpt{i+1}.pth'

                classifier = Classifier(
                    input_size=args.input_size,
                    scale_size=args.scale_size,
                    dropout=args.dropout,
                    train_dir=None,
                    validation_dir=None,
                    model_path=model_path,
                    model_type=args.model_type,
                    device=torch.device(f'cuda:{args.cuda}'),
                    logger=logger
                )

                classifier.eval_model(
                    threshold=args.threshold,
                    data_dir=args.eval_dir
                )

        else:
            classifier = Classifier(
                input_size=args.input_size,
                scale_size=args.scale_size,
                dropout=args.dropout,
                train_dir=None,
                validation_dir=None,
                model_path=args.ckpt,
                model_type=args.model_type,
                device=torch.device(f'cuda:{args.cuda}'),
                logger=logger
            )

            classifier.eval_model(
                threshold=args.threshold,
                data_dir=args.eval_dir
            )


    logger.info('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None, type=str,
                        help="The directory where stores the embeddings and labels of train set")
    parser.add_argument("--validation_dir", type=str,
                        help="The directory where stores the embeddings and labels of validation set")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model's parameters will be stored")

    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="The directory where logs will be written")
    parser.add_argument("--log_file", default=None, type=str, required=True,
                        help="The path where logs will be written")
    parser.add_argument("--cuda", type=int, default=0,
                        help="The cuda where loads the model and data")
    
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--epoch", default=50, type=int,
                        help="The epochs for training")
    parser.add_argument("--save_step", default=5, type=int,
                        help="The step used to save the model ckpt")
    parser.add_argument("--batch_size", default=50, type=int,
                        help="The batch size for training dataset")
    parser.add_argument("--lr", default=0.01, type=float,
                        help="The initial learning rate")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="The dropout rate")
    parser.add_argument("--weight_decay", default=0.00001, type=float,
                        help="The weight decay in Adam")
    parser.add_argument("--lr_step", default=15, type=int,
                        help="The step_size for lr_scheduler.StepLR")
    parser.add_argument("--lr_gamma", default=0.5, type=float,
                        help="The gamma for lr_scheduler.StepLR")
    
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to train the model")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate the trained model on testsets")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="The checkpoint used to evaluate")
    parser.add_argument("--eval_all_ckpt", action="store_true",
                        help="Whether to load all ckpt for evaluation")
    
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['deepseek-coder', 'starcoder2'],
                        help="The type of model")
    parser.add_argument("--input_size", type=int,
                        help="The input size for classifier's MLP")
    parser.add_argument("--scale_size", type=int,
                        help="The scaling size for embeddings")
    
    parser.add_argument("--threshold", default=0.5, type=float,
                        help="Threshold used to device positive and negative samples")

    parser.add_argument("--eval_dir", type=str,
                        help="The directory where stores the embeddings and labels to evaluate")
    
    args = parser.parse_args()

    if args.do_train and args.train_dir is None:
        raise ValueError("Should set train_dir when do_train")
    
    if args.do_train and args.validation_dir is None:
        raise ValueError("Should set validation_dir when do_train")
    
    if args.do_train and args.output_dir is None:
        raise ValueError("Should set output_dir when do_train")

    if args.do_train and args.do_eval:
        raise ValueError("Should not set do_eval when do_train")
    
    if args.do_eval and args.ckpt is None:
        raise ValueError("Please set ckpt when do_eval")
    
    if args.do_eval and args.eval_dir is None:
        raise ValueError("Please set eval_dir when do_eval")

    main(args)