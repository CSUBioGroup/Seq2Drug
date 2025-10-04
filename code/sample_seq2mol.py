import time
import pickle
import argparse
import os, json
import pandas as pd
import torch as th
from tqdm import tqdm
from transformers import set_seed
from dti_datasets import load_data
from diffusion.utils import dist_util, logger
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    load_mol_model,
    add_dict_to_argparser,
    args_to_dict,
    ESMModel
)


def create_argparser():
    defaults = dict(model_path='', molbart_checkpoint='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def write_predictions(path, original_smiles, smiles, log_lhs):
    num_data = len(smiles)
    beam_width = len(smiles[0])
    beam_outputs = [[[]] * num_data for _ in range(beam_width)]
    beam_log_lhs = [[[]] * num_data for _ in range(beam_width)]

    for b_idx, (smiles_beams, log_lhs_beams) in enumerate(zip(smiles, log_lhs)):
        for beam_idx, (smi, log_lhs) in enumerate(zip(smiles_beams, log_lhs_beams)):
            beam_outputs[beam_idx][b_idx] = smi
            beam_log_lhs[beam_idx][b_idx] = log_lhs

    df_data = {
        "original_smiles": original_smiles
    }
    for beam_idx, (outputs, log_lhs) in enumerate(zip(beam_outputs, beam_log_lhs)):
        df_data["prediction_" + str(beam_idx)] = beam_outputs[beam_idx]
        df_data["log_likelihood_" + str(beam_idx)] = beam_log_lhs[beam_idx]

    df = pd.DataFrame(data=df_data).to_csv(path, index=None, sep='\t')
    return df


# def get_prot_inputs(sequences, prot_feats):
#     demo_repr = prot_feats[list(prot_feats.keys())[0]]
#     sequence_repr = th.zeros([len(sequences), 1, demo_repr.shape[-1]], dtype=demo_repr.dtype)
#     for i, seq in enumerate(sequences):
#         sequence_repr[i, 0, :] = prot_feats[seq][0, 0, :]
#     sequence_repr = sequence_repr.to(device=dist_util.dev())
#     return sequence_repr

def get_prot_inputs(sequences, prot_feats):
    demo_repr = prot_feats[sequences[0]]
    maxn = max([prot_feats[seq].shape[1] for seq in sequences])
    sequence_repr = th.zeros([len(sequences), maxn, demo_repr.shape[-1]], dtype=demo_repr.dtype)
    for i, seq in enumerate(sequences):
        repr = prot_feats[seq]
        sequence_repr[i, :repr.shape[1], :] = repr[0, :, :]
    sequence_repr = sequence_repr.to(device=dist_util.dev())
    return sequence_repr


@th.no_grad()
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    mol_model = load_mol_model(args.molbart_checkpoint, device=dist_util.dev())
    max_mol_len = 84

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.eval().requires_grad_(False).to(dist_util.dev())

    set_seed(args.seed2)

    data_test = load_data(
        batch_size=args.batch_size,
        data_args=args,
        split=args.split,
        loop=False
    )

    prot_feats = pickle.load(open(args.data_dir + '/prot_feats_' + str(args.seq_len) + '.pkl', 'rb'))

    start_t = time.time()

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.csv")

    print("### Sampling...on", args.split)
    ori_smiles = []
    gen_smiles = []
    gen_log_lhs = []
    for batch in tqdm(data_test):

        sequences = batch['target_sequence']
        prot_encodes = get_prot_inputs(sequences, prot_feats)
        prot_inputs = model.prot_layer(prot_encodes)
        
        x_start = th.zeros((prot_inputs.shape[0], prot_inputs.shape[1]+max_mol_len, prot_inputs.shape[2]), dtype=prot_inputs.dtype, device=prot_inputs.device)
        x_start[:, :prot_inputs.shape[1], :] = prot_inputs[:, :, :]

        input_mask = th.full([x_start.shape[0], x_start.shape[1]], 0, dtype=th.int64)
        input_mask[:, prot_inputs.shape[1]:] = 1

        input_mask_ori = input_mask

        noise = th.randn_like(x_start)
        input_mask = th.broadcast_to(input_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_mask == 0, x_start, noise)

        model_kwargs = {}

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample_shape = (x_start.shape[0], x_start.shape[1], x_start.shape[2])
        samples = sample_fn(
            model,
            sample_shape,
            noise=x_noised,
            clip_denoised=args.clip_denoised,
            # denoised_fn=partial(denoised_fn_round, args, model_emb),
            denoised_fn=None,
            model_kwargs=model_kwargs,
            top_p=args.top_p,
            clamp_step=args.clamp_step,
            clamp_first=True,
            mask=input_mask,
            x_start=x_start,
            gap=step_gap
        )

        sample = samples[-1]

        unmask_len = int(sum(input_mask_ori[0] == 0))
        mol_diff = sample[:, unmask_len:, :]
        memory = model.mol_out_layer(mol_diff).transpose(0, 1)
        mask = [[False] * (max_mol_len) for _ in range(sample.shape[0])]
        mem_pad_mask = th.tensor(mask, dtype=th.bool, device=memory.device).transpose(0, 1)
        mol_strs, log_lhs = mol_model.sample_molecules(memory, mem_pad_mask, sampling_alg="beam")
        
        ori_smiles.extend(batch['SMILES'])
        gen_smiles.extend(mol_strs)
        gen_log_lhs.extend(log_lhs)
    
    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')
    write_predictions(out_path, ori_smiles, gen_smiles, gen_log_lhs)


if __name__ == "__main__":
    main()
