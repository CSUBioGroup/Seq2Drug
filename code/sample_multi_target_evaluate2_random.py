import json
import time
import pickle
import argparse
import torch
import torch as th
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolLogP, qed
from tqdm.auto import tqdm

from evaluation.docking import *
from evaluation.docking_2 import *
from evaluation.sascorer import *
from evaluation.score_func import *
from evaluation.similarity import calculate_diversity

from tqdm import tqdm
from transformers import set_seed
from dti_datasets import load_data
from diffusion.step_sample import create_named_schedule_sampler
from diffusion.utils import dist_util
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    load_mol_model,
    add_dict_to_argparser,
    args_to_dict
)
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class MolReconsError(Exception):
    pass


def get_prot_inputs(sequences, prot_feats):
    demo_repr = prot_feats[sequences[0]]
    maxn = max([prot_feats[seq].shape[1] for seq in sequences])
    sequence_repr = th.zeros([len(sequences), maxn, demo_repr.shape[-1]], dtype=demo_repr.dtype)
    for i, seq in enumerate(sequences):
        repr = prot_feats[seq]
        sequence_repr[i, :repr.shape[1], :] = repr[0, :, :]
    sequence_repr = sequence_repr.to(device=dist_util.dev())
    return sequence_repr


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def load_training_config():
    
    with open(json_file, 'r') as f:
        return json.load(f)


def create_argparser():
    defaults = dict(model_path='', molbart_checkpoint='', step=2000, out_dir='', top_p=-1)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_training_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def save_sdf(mol, sdf_dir, gen_file_name):
    writer = Chem.SDWriter(os.path.join(sdf_dir, gen_file_name))
    writer.write(mol, confId=0)
    writer.close()

def save_smiles(smi, smi_dir, smi_name):
    with open(os.path.join(smi_dir, smi_name), 'w') as f:
        f.write(smi)

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def smiles2mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_with_H = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_H)
    AllChem.MMFFOptimizeMolecule(mol_with_H)
    embedded_mol = Chem.RemoveHs(mol_with_H)
    return embedded_mol


def evaluate(mol, receptor_file):
    _, sa = compute_sa_score(mol)
    _qed = qed(mol)
    logP = MolLogP(mol)
    Lipinski = obey_lipinski(mol)
    vina_score = calculate_qvina2_score(receptor_file, mol, sdf_dir)[0]
    return sa, _qed, logP, Lipinski, vina_score


# json_file = 'models/seq2mol/seq2mol_decodeloss_crossdocked_pocket10_combined_large_f1_l128_h128_lr0.0001_t2000_sqrt_lossaware_seed102_data_crossdocked_pocket1020240625-11:08:36/training_args.json'
# args = create_argparser().parse_args()
# model_n = 'crossdocked_decodeloss2000'
# mol_model_path = args.checkpoint_path + '/mol_model_002000.pth'
# model_path = args.checkpoint_path + '/ema_0.9999_002000.pt'

# json_file = 'models/seq2mol/seq2mol_mse-tT-tokenloss_crossdocked_pocket10_combined_large_f0_l128_h128_lr0.0001_t2000_sqrt_lossaware_seed102_data_crossdocked_pocket1020240705-13:23:48/training_args.json'
# args = create_argparser().parse_args()
# model_n = 'crossdocked_mse-tT-tokenloss4000'
# mol_model_path = args.checkpoint_path + '/mol_model_004000.pth'
# model_path = args.checkpoint_path + '/ema_0.9999_004000.pt'

json_file = 'models/seq2mol/seq2mol_tokenloss_crossdocked_pocket10_combined_large_f0_l128_h128_lr0.0001_t2000_sqrt_lossaware_seed102_data_crossdocked_pocket1020240710-21:19:07/training_args.json'
args = create_argparser().parse_args()
model_n = 'crossdocked_tokenloss6000'
mol_model_path = args.checkpoint_path + '/mol_model_006000.pth'
model_path = args.checkpoint_path + '/ema_0.9999_006000.pt'

set_seed(args.seed)
dist_util.setup_dist()

# mol_model_path = '../demo3/models/seq2mol/seq2mol_BindingDB_combined_large_f1_h128_lr0.0001_t2000_sqrt_lossaware_seed102_data_BindingDB20230707-00:44:15/mol_model_002000.pth'
# mol_model = MolBartModel(model_path=args.molbart_path, device=dist_util.dev())  # 原始模型
mol_model = load_mol_model(mol_model_path, device=dist_util.dev())  # 保存模型

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, load_defaults_config().keys())
)

model.load_state_dict(
    dist_util.load_state_dict(model_path, map_location="cpu")
)

schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'### The parameter count is {pytorch_total_params}')

model.eval().requires_grad_(False).to(dist_util.dev())
set_seed(123)

args.data_dir = '../datasets/multi_target'
data_test = load_data(
    batch_size=1,
    data_args=args,
    split='test',
    loop=False
)

data_seq = pd.read_csv(args.data_dir + "/test_interactions.csv")[["target_sequence"]]
# from protein_feature_process import get_protein_feature
# prot_feats = get_protein_feature(data_seq)
with open(args.data_dir + '/test_prot_feats_128.pkl', 'rb') as f:
    prot_feats = pickle.load(f)

max_mol_len = 84
# index = -64

time_list = []

r_sa_list = []
r_qed_list = []
r_logP_list = []
r_Linpinski_list = []
r_vina_score_list = []

sa_list = []
qed_list = []
logP_list = []
Lipinski_list = []
vina_score_list = []
smile_list = []
mol_list = []

results = []
protein_files = []

data_path = args.data_dir + '/test_data'
sampling_alg = 'beam'
res_path = './results_random/multitarget-1000/' + '_'.join([model_n, sampling_alg, 'forward', 'random-1-200']) + '/'
sdf_dir = res_path + 'gen_sdf'

# with open(os.path.join(data_path, 'test_vina_crossdock.pkl'), 'rb') as f:
#     test_vina_score_list = pickle.load(f)

test_pairs = ['MEK1/mTOR', 'PARP1/BRD4', 'CDK7/PRMT5', 'CDK9/PRMT5', 'CDK12/PRMT5', 'BRD4/ERBB2', 'BRD4/FGFR3', 'BRD4/TOP1', 'ERBB2/TOP1', 'TOP1/FGFR3']

id_dic = {
    'MEK1': '7m0y',
    'mTOR': '3fap',
    'PARP1': '7kk4',
    'BRD4': '3mxf',
    'CDK7': '6xd3',
    'CDK9': '6z45',
    'CDK12': '7nxk',
    'PRMT5': '6rlq',
    'ERBB2': '7pcd',
    'FGFR3': '6lvm',
    'TOP1': '1tl8'
}


for i, batch in enumerate(tqdm(data_test)):

    if i>=2 :
        break

    t1, t2 = test_pairs[i].split('/')
    id1, id2 = id_dic[t1], id_dic[t2]

    receptor_file1 = Path(os.path.join(data_path, id1 + '_pocket.pdbqt'))
    r_sdf_file1 = os.path.join(data_path, id1 + '_ligand.sdf')
    r_mol1 = Chem.SDMolSupplier(r_sdf_file1, sanitize=False)[0]
    r_smiles1 = mol2smiles(r_mol1)

    receptor_file2 = Path(os.path.join(data_path, id2 + '_pocket.pdbqt'))
    r_sdf_file2 = os.path.join(data_path, id2 + '_ligand.sdf')
    r_mol2 = Chem.SDMolSupplier(r_sdf_file2, sanitize=False)[0]
    r_smiles2 = mol2smiles(r_mol2)

    r_sa1, r_qed1, r_logP1, r_Lipinski1, r_vina_score1 = evaluate(r_mol1, receptor_file1)
    r_sa2, r_qed2, r_logP2, r_Lipinski2, r_vina_score2 = evaluate(r_mol2, receptor_file2)

    print("Reference SA score:", r_sa1, r_sa2)
    print("Reference QED score:", r_qed1, r_qed2)
    print("Reference logP:", r_logP1, r_logP2)
    print("Reference Lipinski:", r_Lipinski1, r_Lipinski2)
    print("Reference Vina score:", r_vina_score1, r_vina_score2)

    r_sa_list.append([r_sa1, r_sa2])
    r_qed_list.append([r_qed1, r_qed2])
    r_logP_list.append([r_logP1, r_logP2])
    r_Linpinski_list.append([r_Lipinski1, r_Lipinski2])
    r_vina_score_list.append([r_vina_score1, r_vina_score2])

    sequences = batch['target_sequence']
    prot_encodes = get_prot_inputs(sequences, prot_feats)
    prot_inputs = model.prot_layer(prot_encodes)
    
    num_samples = 1000
    generated_smiles = set()
    t_start = time.time()
    while num_samples > 0:

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

        import random
        index = random.randint(1, 200)

        sample = samples[-index]
        unmask_len = int(sum(input_mask_ori[0] == 0))
        mol_diff = sample[:, unmask_len:, :]
        memory = model.mol_out_layer(mol_diff).transpose(0, 1)
        mask = [[False] * (max_mol_len) for _ in range(sample.shape[0])]
        mem_pad_mask = th.tensor(mask, dtype=th.bool, device=memory.device).transpose(0, 1)

        mol_strs, log_lhs = mol_model.sample_molecules(memory, mem_pad_mask, sampling_alg=sampling_alg)
        if sampling_alg == 'beam':
            mol_strs = mol_strs[0]

        for g_smiles in mol_strs:
            try:
                g_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(g_smiles), canonical=True)
                print(g_smiles)
                if g_smiles in generated_smiles:
                    print('Repeating molecules')
                    raise MolReconsError()
                else:
                    generated_smiles.add(g_smiles)
                if g_smiles is None:
                    print('Error molecules')
                    raise MolReconsError()
                if g_smiles.count('.') > 0:
                    print('Unstable molecules')
                    raise MolReconsError()
                if len(g_smiles) < 4:
                    print('Small molecules')
                    raise MolReconsError()
                
                # if len(generated_smiles) > 500:
                #     mol_frags = Chem.rdmolops.GetMolFrags(gmol, asMols=True, sanitizeFrags=False)
                #     gmol = max(mol_frags, default=gmol, key=lambda m: m.GetNumAtoms())
                #     g_smile = Chem.MolToSmiles(gmol)
                #     print("largest generated smile part:", g_smile)
                #     if g_smile is None:
                #         raise MolReconsError()
                
                gmol = smiles2mol(g_smiles)
                _, g_sa = compute_sa_score(gmol)
                print("Generate SA score:", g_sa)

                g_qed = qed(gmol)
                print("Generate QED score:", g_qed)

                g_logP = MolLogP(gmol)
                print("Generate LogP score:", g_logP)

                g_Lipinski = obey_lipinski(gmol)
                print("Generate Lipinski score:", g_Lipinski)

                gen_index = 1000-num_samples
                g_vina_score1 = calculate_qvina2_score(receptor_file1, gmol, sdf_dir, index=gen_index, r_mol=r_mol1)[0]
                if g_vina_score1 > -2:
                    raise MolReconsError()
                print("Generate Vina score:", g_vina_score1)
                print("Reference Vina score:", r_vina_score1)

                g_vina_score2 = calculate_qvina2_score(receptor_file2, gmol, sdf_dir, index=gen_index, r_mol=r_mol2)[0]
                if g_vina_score2 > -2:
                    raise MolReconsError()
                print("Generate Vina score:", g_vina_score2)
                print("Reference Vina score:", r_vina_score2)

                name = id1 + '_' + id2
                gen_file_name = name + '_gen' + str(gen_index) + '.sdf'
                save_sdf(gmol, sdf_dir, gen_file_name)

                gen_file_smi = name + '_gen' + str(gen_index) + '.smiles'
                save_smiles(g_smiles, sdf_dir, gen_file_smi)
                
                sa_list.append(g_sa)
                qed_list.append(g_qed)
                logP_list.append(g_logP)
                Lipinski_list.append(g_Lipinski)
                vina_score_list.append([g_vina_score1, g_vina_score2])

                metrics = {'SA':g_sa,'QED':g_qed,'logP':g_logP,'Lipinski':g_Lipinski,
                        'vina':[g_vina_score1, g_vina_score2]}
                result = {'smile': g_smiles,
                        'pair': test_pairs[i],
                        'generated_ligand_sdf': gen_file_name,
                        'generated_mol': gmol,
                        'metric_result':metrics
                        }
                results.append(result)

                mol_list.append(gmol)
                smile_list.append(g_smiles)
                num_samples -= 1

                print('Successfully generate molecule for {}, remining {} samples generated'.format(name, num_samples))
            
            except:
                print('Invalid, continue')
        
            if num_samples == 0:
                break
    
    time_list.append(time.time() - t_start)
    print(name + 'takes {} seconds'.format(time.time() - t_start))

times_arr = torch.tensor(time_list)
print(f"Time per pocket: {times_arr.mean():.3f} \pm "
      f"{times_arr.std(unbiased=False):.2f}")

save_path = res_path + 'samples_all.pkl'
print('Saving samples to: %s' % save_path)
with open(save_path, 'wb') as f:
    pickle.dump(results, f)
    f.close()

all_results = {
    'time_list': time_list,
    'r_sa_list': r_sa_list,
    'r_qed_list': r_qed_list,
    'r_logP_list': r_logP_list,
    'r_Linpinski_list': r_Linpinski_list,
    'r_vina_score_list': r_vina_score_list,
    'sa_list': sa_list,
    'qed_list': qed_list,
    'logP_list': logP_list,
    'Lipinski_list': Lipinski_list,
    'vina_score_list': vina_score_list,
    'smile_list': smile_list,
    'mol_list': mol_list
}
all_results_save_path = res_path + 'all_results.pkl'
print('Saving all results to: %s' % all_results_save_path)
with open(all_results_save_path, 'wb') as f:
    pickle.dump(all_results, f)
    f.close()

print('generate:%d' % len(smile_list))

print('reference mean sa: {}, num: {}'.format(np.mean(r_sa_list), len(r_sa_list)))
print('reference mean qed: {}, num: {}'.format(np.mean(r_qed_list), len(r_qed_list)))
print('reference mean logP: {}, num: {}'.format(np.mean(r_logP_list), len(r_logP_list)))
print('reference mean Lipinski: {}, num: {}'.format(np.mean(r_Linpinski_list), len(r_Linpinski_list)))
print('reference reference mean vina: {}, num: {}'.format(np.mean(r_vina_score_list, axis=0), len(r_vina_score_list)))

print('mean sa: {}, num: {}'.format(np.mean(sa_list), len(sa_list)))
print('mean qed: {}, num: {}'.format(np.mean(qed_list), len(qed_list)))
print('mean logP: {}, num: {}'.format(np.mean(logP_list), len(logP_list)))
print('mean Lipinski {}, num: {}'.format(np.mean(Lipinski_list), len(Lipinski_list)))
print('mean vina: {}, num: {}'.format(np.mean(vina_score_list, axis=0), len(vina_score_list)))

print('diversity:%f' % calculate_diversity(mol_list)) 
