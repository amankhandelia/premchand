import os
import torch
import json

from network import PremchandLanguageModel
from dataset import LanguageModelDataset
from train import LMTrainer

from torch.nn import NLLLoss
from torch.optim import Adam



def main(run, log_dir, weight_dir, tensor_path, seq_len=32, emb_sz = 128, hidden_size = 128, lr = 0.0001, epochs = 100, vocab_sz = 10000, verbose = True):
    path_to_logs = os.path.join(log_dir, run)
    path_to_weight_file = os.path.join(weight_dir, run)
    dataset = LanguageModelDataset.from_tensor(tensor_path, seq_len=seq_len, vocab_size=vocab_sz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PremchandLanguageModel(vocab_size = dataset.vocab.vocab_size, embedding_dim=emb_sz, hidden_dim=hidden_size).to(device)
    trainer = LMTrainer(model=model, dataset=dataset, criterion=NLLLoss())
    optimizer = Adam(model.parameters(), lr=lr)
    trainer.train(optimizer=optimizer, path_to_logs=path_to_logs, path_to_weight_file=path_to_weight_file, device=device, verbose=verbose, epochs=epochs)

if __name__ == '__main__':
    props = json.load(open('static.json'))
    run = props['run']
    experiment_name = props['experiment_name']
    exp_dir = props['exp_dir'].format(experiment_name)
    logs = props['logs'].format(exp_dir)
    weights = props['weights'].format(exp_dir)
    tensor = props['tensor']
    main(run=run, log_dir=logs, weight_dir=weights, tensor_path=tensor)