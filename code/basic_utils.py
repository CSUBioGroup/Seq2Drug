import argparse
import torch
import torch.nn as nn
import json
import esm
import molbart.util as mol_util
from molbart.decoder import DecodeSampler
from functools import partial

from diffusion import gaussian_diffusion as gd
from diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffusion.transformer_model import TransformerNetModel


class ESMModel(nn.Module):

    def __init__(self, model_name='esm2_t33_650M_UR50D', device='cuda'):

        super().__init__()

        if model_name == 'esm2_t36_3B_UR50D':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        
        elif model_name == 'esm2_t33_650M_UR50D':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        
        elif model_name == 'esm2_t30_150M_UR50D':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        
        else:
            assert False, "invalid model name"
        
        self.device = device
        self.model = model.to(self.device)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()
        self.repr_layer = self.model.num_layers

    def batch_encode(self, sequences, max_length=1022):

        data = [(f'protein_{i}', seq[:max_length]) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
        
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
        token_repr = results["representations"][self.repr_layer]

        return token_repr


class MolBartModel(nn.Module):

    def __init__(self, model_path, num_beams=10, device='cuda'):

        super().__init__()

        self.device = device
        self.tokeniser = mol_util.load_tokeniser(mol_util.DEFAULT_VOCAB_PATH, mol_util.DEFAULT_CHEM_TOKEN_START)
        sampler = DecodeSampler(self.tokeniser, mol_util.DEFAULT_MAX_SEQ_LEN)
        self.model = mol_util.load_bart(model_path, sampler)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.num_beams = num_beams
        sampler.max_seq_len = self.model.max_seq_len
    
    def batch_encode(self, smiles):
        batch = self._tokenise(smiles)
        device_batch = {
            key: val.to(self.device) if type(val) == torch.Tensor else val for key, val in batch.items()
        }
        
        with torch.no_grad():
            memory = self.model.encode(device_batch)
            mem_mask = device_batch["encoder_pad_mask"].clone()
        
        device_batch["memory_input"] = memory
        device_batch["memory_pad_mask"] = mem_mask
        
        return device_batch
    
    def batch_decode(self, batch):
        
        memory_input = batch["memory_input_"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_input = batch["encoder_input"].clone()[:-1, :]
        decoder_pad_mask = batch["encoder_pad_mask"].clone()[:-1, :].transpose(0, 1)
        
        decoder_embs = self.model._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self.model._generate_square_subsequent_mask(seq_len, device=decoder_embs.device)

        model_output = self.model.decoder(
            decoder_embs, 
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask
        )
        token_output = self.model.token_fc(model_output)
        token_probs = self.model.log_softmax(token_output)
        
        output = {
            "model_output": model_output,
            "token_output": token_output,
            "token_probs": token_probs
        }

        return output
    
    def sample_molecules(self, memory, mem_mask, sampling_alg="greedy"):
        
        self.model.freeze()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self.model._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.model.sampler.greedy_decode(decode_fn, batch_size, memory.device)

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.model.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.model.num_beams)

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        self.model.unfreeze()

        return mol_strs, log_lhs
    
    def _tokenise(self, smiles):
        tokenised = self.tokeniser.tokenise(smiles, pad=True)

        tokens = tokenised["original_tokens"]
        mask = tokenised["original_pad_masks"]
        tokens, mask = self._check_seq_len(tokens, mask)

        token_ids = self.tokeniser.convert_tokens_to_ids(tokens)

        token_ids = torch.tensor(token_ids).transpose(0, 1)
        pad_mask = torch.tensor(mask, dtype=torch.bool).transpose(0, 1)

        output = {
            "encoder_input": token_ids,
            "encoder_pad_mask": pad_mask
        }

        return output
    
    def _check_seq_len(self, tokens, mask):
        seq_len = max([len(ts) for ts in tokens])
        max_seq_len = self.model.max_seq_len
        if seq_len > max_seq_len:
            tokens_short = [ts[:max_seq_len] for ts in tokens]
            mask_short = [ms[:max_seq_len] for ms in mask]
            return tokens_short, mask_short
        return tokens, mask
    
    def _calc_loss(self, batch_input, model_output):

        tokens = batch_input["encoder_input"].clone()[1:, :].transpose(0, 1)
        pad_mask = batch_input["encoder_pad_mask"].clone()[1:, :].transpose(0, 1)
        token_output = model_output["token_output"].transpose(0, 1)

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        loss_all = self.model.loss_fn(token_output.transpose(1, 2), target)
        num_tokens = (~target_mask).sum(dim=-1)
        loss = loss_all.sum(dim=-1) / num_tokens
        return loss

def load_mol_model(path, device):
    print("Load molbart model from: ", path)
    mol_model = torch.load(path, map_location=torch.device('cpu'))
    mol_model.device = device
    mol_model.model.to(device)
    mol_model.model.eval()
    return mol_model


def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    mol_dim,
    prot_dim,
    hidden_t_dim,
    hidden_dim,
    config_name,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        mol_dims=mol_dim,
        prot_dims=prot_dim,
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
