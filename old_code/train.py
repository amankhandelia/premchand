import torch
import os
import random

from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter

class LMTrainer:
    def __init__(self, model, dataset, criterion):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion

    @staticmethod
    def collate(items):
        """
        Collate all the tensors into a batch
        """
        sources = []
        targets = []
        for item in items:
            sources.append(item[0])
            targets.append(item[1])
        sources = torch.stack(sources)
        targets = torch.stack(targets)
        return sources, targets

    def train(self, optimizer, path_to_weight_file=None, path_to_logs=None, epochs=100, batch_size=32, val_pct=0.20,
              device='cpu', verbose=False, seed=42):
        """
        Prepares the validation and train subset, by randomly shuffling the dataset, one can fix that using a seed.
        It trains the model on the data, for each epoch it persists the weight, logs the metrics and for each batch logs the loss.
        If you wish to see what is happening, turn verbose on.

        @param optimizer[torch.optim]: A torch.optim object instance
        @param path_to_weight_file[str]: Folder to which the weight files will be persisted
        @param path_to_logs[str]: Folder to which Tensorboard log file will be persisted
        @param epochs[int]: Number of epochs for which to run the training
        @param batch_size[int]: Number of training instance for each batch
        @param val_pct[int]: Percentage of dataset against whom validation (logging of metrics) will be performed
        @param device[torch.Device]: Device on which model and data should reside
        @param verbose[boolean]: Print details of inner working on the console
        @param seed[int]: To control randomness

        @return: None
        """
        random.seed(seed)

        if verbose:
            print("Preparing for training")

        # indices for random split of dataset into train and validation subset
        indices = list(range(len(self.dataset)))
        shuffle = False
        if shuffle:
            random.shuffle(indices)

        # using sorted here because we want sequence to be ordered by length when training them, taking a page out of Curriculum Learning et al Yoshua Bengio
        val_subset = Subset(self.dataset, indices[:int(len(indices) * val_pct)])
        train_subset = Subset(self.dataset, indices[int(len(indices) * val_pct):])

        # getting all the loaders ready for train and validation
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, collate_fn=self.collate, drop_last=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size // 2, collate_fn=self.collate, drop_last=False)

        # getting the tenosrboard writer ready
        if path_to_logs is None:
            summary_writer = SummaryWriter()
        else:
            summary_writer = SummaryWriter(path_to_logs)

        # restart the training if required
        start_epoch = self.load_last_model(path_to_weight_file, verbose=verbose)
        i = start_epoch * (len(train_subset) // batch_size)

        if verbose:
            print("Preparation Done. Training Started")

        # training starts
        for epoch in range(start_epoch, epochs):
            j = 0
            self.model.train()
            for x, y in train_dataloader:
                optimizer.zero_grad()
                y_dash = self.model(x.to(device))
                y_dash = y_dash.reshape(-1, y_dash.size(-1))
                loss = self.criterion(y_dash, y.view(-1).to(device))
                summary_writer.add_scalar('loss', loss, i)
                ppl = torch.exp(loss / y_dash.size(0))
                summary_writer.add_scalar('perplexity', ppl, i)
                loss.backward()
                optimizer.step()
                i += 1
                j += 1
            self.log_metrics(val_dataloader, summary_writer, epoch, verbose=verbose, device=device)
            self.save_model(path_to_weight_file, epoch)
            if verbose:
                print("Epoch:{}. Completed".format(epoch))

    def log_metrics(self, dataloader, writer, epoch_count, device='cpu', verbose=False):
        """
        Log metrics to the Tensorboard on validation dataset.

        @param dataloader: Dataloader linked to the validation dataset
        @param writer: SummaryWriter object for writing tensorboard logs
        @param epoch_count: Epoch at which the metrics are being logged
        @param device: Device on which forward propagation will be performed
        @param verbose: Boolean value, to be self-descriptive or not

        @return: None
        """
        if verbose:
            print("Started running test phase for Epoch: {}".format(epoch_count))

        num_examples_to_summary = 2
        validation_subset = dataloader.dataset
        dataset = validation_subset.dataset
        self.model.eval()
        j = 0
        steps_so_far = epoch_count * (len(validation_subset) // dataloader.batch_size)
        cum_loss, cum_ppl = 0, 0

        for x, y in dataloader:
            global_timestep = steps_so_far + j
            # forward pass of the model, to get the y_dash on the validation set
            with torch.no_grad():
                y_dash = self.model(x.to(device))
                y_dash = y_dash.reshape(-1, y_dash.size(-1))
                loss = self.criterion(y_dash, y.view(-1).to(device))
                ppl = torch.exp(loss)
                cum_loss += loss
                cum_ppl += ppl
                j += 1
        # add validation loss to the logs (tensorboad)
        writer.add_scalar('val perplexity', cum_ppl / j, global_timestep)
        writer.add_scalar('val loss', cum_loss / j, global_timestep)

    def save_model(self, path_to_weight_file, epoch_count, verbose=False):
        """
        For persisting model at any given epoch. It persists model to the given path_to_weight_file.
        Filename depends on epoch_count parameter.

        @param path_to_weight_file[str]: Folder where we will persist the weight files for any given run
        @param epoch_count[int]: Epoch at which the weight file is to be persisted
        @param verbose[boolean]: To be self-descriptive or not

        @return: None
        """
        if not os.path.exists(path_to_weight_file):
            os.makedirs(path_to_weight_file)
        filename = os.path.join(path_to_weight_file, 'model_00{}.pth'.format(epoch_count))
        torch.save(self.model.state_dict(), filename)
        if verbose:
            print('Weight file for epoch {} persisted at: {}'.format(epoch_count, filename))

    def load_last_model(self, weight_file_path, verbose=False):
        """
        To restart the training from where we left. If present, it reads the last model serialized
        otherwise does nothing

        @param weight_file_path[str]: Path to the folder which contains the weight files for any given run
        @param verbose[boolean]: To be self-descriptive or not

        @return: 0 if no such model is there and if loaded, its epoch count
        """
        if verbose:
            print("Checking if there is any model to load")
        if os.path.exists(weight_file_path):
            weight_files_index = sorted([(item, int(item[6:-4])) for item in os.listdir(weight_file_path)],
                                        key=lambda k: k[1])
            if len(weight_files_index) > 0:
                self.model.load_state_dict(torch.load(os.path.join(weight_file_path, weight_files_index[-1][0])))
                if verbose:
                    print("Loaded model at epoch {} from path {}".format(weight_files_index[-1][1],
                                                                         os.path.join(weight_file_path,
                                                                                      weight_files_index[-1][0])))
                return weight_files_index[-1][1] + 1
            else:
                if verbose:
                    print("Couldn't find a model to load")
                return 0
        else:
            if verbose:
                print('No model present to load')
            return 0