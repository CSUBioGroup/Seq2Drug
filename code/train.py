import argparse
import json, os
import time
import pickle
from diffusion.utils import dist_util, logger
from diffusion.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    MolBartModel
)
from dti_datasets import load_data
from train_util import TrainLoop
from transformers import set_seed
import wandb

# os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_MODE"] = "offline"

def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")
    
    data = load_data(
        batch_size=args.batch_size,
        data_args=args,
        split='train'
    )

    data_val = load_data(
        batch_size=args.batch_size,
        data_args=args,
        split='valid'
    )

    prot_feats = pickle.load(open(args.data_dir + '/prot_feats_' + str(args.seq_len) + '.pkl', 'rb'))
    
    logger.log("### Creating model and diffusion...")
    mol_model = MolBartModel(model_path=args.molbart_path, device=dist_util.dev())
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    model.to(dist_util.dev())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "Seq2Mol"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")
    start_t = time.time()
    TrainLoop(
        mol_model=mol_model,
        prot_feats=prot_feats,
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_val,
        eval_interval=args.eval_interval,
        fine_tune=bool(args.fine_tune)
    ).run_loop()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))

if __name__ == "__main__":
    main()
