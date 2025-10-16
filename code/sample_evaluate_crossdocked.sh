python -m torch.distributed.launch --nproc_per_node=1 --master_port=12233 --use_env --use_env sample_evaluate_crossdocked.py \
--model_path models/seq2drug/seq2drug_crossdocked \
--model_checkpoint model.pt \
--mol_model_checkpoint mol_model.pth
