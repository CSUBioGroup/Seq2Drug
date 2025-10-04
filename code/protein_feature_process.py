import pandas as pd
import pickle
import torch
from tqdm import tqdm
from basic_utils import ESMModel


def get_protein_feature(data, maxl=128):
    
    prot_model = ESMModel(device="cuda")
    seq_feats = {}
    data = data.drop_duplicates(subset=['target_sequence'])
    for seq in tqdm(list(data['target_sequence'])):
        seq = seq.upper()
        seq_feats[seq] = prot_model.batch_encode([seq], max_length=maxl).cpu()
    
    return seq_feats


def get_protein_feature4multi_target(data, maxl=128):
    
    prot_model = ESMModel(device="cuda")
    seq_feats = {}
    data = data.drop_duplicates(subset=['target_sequence'])
    for sequence in tqdm(list(data['target_sequence'])):
        seqs = sequence.split('+')
        n = len(seqs)
        l = maxl // n
        seq_combine = ''
        for s in seqs:
            seq_combine += s[:l]
        seq_feats[sequence] = prot_model.batch_encode([seq_combine], max_length=maxl).cpu()
    
    return seq_feats
    

if __name__ == "__main__":

    maxl = 128

    data_dir = "../datasets/crossdocked_pocket10/"
    data_test = pd.read_csv(data_dir + "test_interactions.csv")[["target_sequence"]]

    seq_feats = get_protein_feature(data_test, maxl=maxl)
    with open(data_dir + 'test_prot_feats_' + str(maxl) + '.pkl', 'wb') as f:
        pickle.dump(seq_feats, f)


    # data_dir = "../datasets/multi_target/"
    # data_test = pd.read_csv(data_dir + "test_interactions.csv")[["target_sequence"]]

    # seq_feats = get_protein_feature4multi_target(data_test, maxl=maxl)
    # with open(data_dir + 'test_prot_feats_' + str(maxl) + '.pkl', 'wb') as f:
    #     pickle.dump(seq_feats, f)
