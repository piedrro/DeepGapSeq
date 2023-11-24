import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from skimage import exposure
from datetime import datetime
import os
import pathlib
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import copy
import warnings


class Trainer:

    def __init__(self,
            model: torch.nn.Module = None,
            pretrained_model=None,
            device: torch.device = None,
            criterion: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None,
            trainloader: torch.utils.data.Dataset = None,
            valoader: torch.utils.data.Dataset = None,
            batch_size: int = None,
            lr_scheduler: torch.optim.lr_scheduler = None,
            tensorboard=bool,
            epochs: int = 100,
            model_folder='',
            model_path=None,
            save_dir='',
            timestamp=datetime.now().strftime("%y%m%d_%H%M"), ):


        self.model = model
        self.pretrained_model = pretrained_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.trainloader = trainloader
        self.valoader = valoader
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model_folder = model_folder
        self.save_dir = save_dir
        self.model_path = model_path
        self.epoch = 0
        self.tensorboard = tensorboard
        self.num_train_images = len(self.trainloader) * self.batch_size
        self.num_validation_images = len(self.valoader) * self.batch_size
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rate = []
        self.timestamp = timestamp
        self.best_epoch = 0
        self.best_model_weights = None

        if os.path.exists(save_dir):
            model_dir = os.path.join(save_dir, "deepgapseq_simulated_traces/models", model_folder + "_" + self.timestamp)
        else:

            model_dir = os.path.join("deepgapseq_simulated_traces/models", model_folder + "_" + self.timestamp)

        model_dir = os.path.abspath(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if pretrained_model:
            if os.path.isfile(pretrained_model):
                model_weights = torch.load(os.path.abspath(pretrained_model))['model_state_dict']
                model.load_state_dict(model_weights)

        # if tensorboard:
        #     self.writer = SummaryWriter(log_dir="runs/" + self.model_folder + "_" + timestamp)

        self.model_path = os.path.join(model_dir, f"inceptiontime_model_{self.timestamp}")

    def calculate_accuracy(self, preds, labels):

        # Assuming preds and labels are of shape [batch_size, num_classes, sequence_length]
        # Convert probabilities to binary predictions
        preds_binary = preds.argmax(dim=1)  # Get the indices of the max values
        labels_binary = labels.argmax(dim=1)
    
        correct = (preds_binary == labels_binary).float()  # Convert to float for division
        accuracy = correct.mean()  # Calculate mean
    
        return accuracy.item()

    def train(self):

        progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=True)

        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self.train_step()
            
            # break

            # """Validation block"""
            # if self.valoader is not None:
            #     self.val_step()
            #
            # """update tensorboard"""
            # if self.writer:
            #     self.writer.add_scalar("Loss/train", self.training_loss[-1], self.epoch)
            #     self.writer.add_scalar("Loss/validation", self.validation_loss[-1], self.epoch)
            #     self.writer.add_scalar("Accuracy/train", self.training_accuracy[-1], self.epoch)
            #     self.writer.add_scalar("Accuracy/validation", self.validation_accuracy[-1], self.epoch)
            #
            # """Learning rate scheduler block"""
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()  # learning rate scheduler step
            #
            # if self.validation_loss[-1] == np.min(self.validation_loss):
            #     self.best_epoch = self.epoch
            #     self.best_model_weights = self.model.state_dict()
            #
            # torch.save({'best_epoch': self.best_epoch, 'num_epochs': self.epochs, 'model_state_dict': self.best_model_weights, 'optimizer_state_dict': self.optimizer.state_dict(), 'lr_scheduler': self.lr_scheduler, 'training_loss': self.training_loss, 'validation_loss': self.validation_loss, 'training_accuracy': self.training_accuracy, 'validation_accuracy': self.validation_accuracy, 'num_validation_images': self.num_validation_images}, self.model_path)
            #
            # progressbar.set_description(f'(Training Loss {self.training_loss[-1]:.5f}, Validation Loss {self.validation_loss[-1]:.5f})')  # update progressbar

        return self.model_path, self.best_model_weights


    def train_step(self):

        train_losses = []  # accumulate the losses here
        train_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=1, leave=True)

        for i, (data, labels) in batch_iter:
            data, labels = data.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)
            
            self.optimizer.zero_grad()  # zerograd the parameters
            pred_labels = self.model(data)  # one forward pass
        
            loss = self.criterion(pred_labels, labels)
            train_losses.append(loss.item())

            accuracy = self.calculate_accuracy(pred_labels, labels)
            train_accuracies.append(accuracy)

            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            current_lr = self.optimizer.param_groups[0]['lr']

            batch_iter.set_description(f'Training: (loss {np.mean(train_losses):.5f}, Acc {np.mean(train_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(train_accuracies))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    # def val_step(self):

    #     self.model.eval()  # evaluation mode
    #     valid_losses = []  # accumulate the losses here
    #     valid_accuracies = []

    #     batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=1, leave=True)

    #     for i, (data, labels) in batch_iter:
    #         data, labels = data.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

    #         with torch.no_grad():

    #             self.optimizer.zero_grad()  # zerograd the parameters
    #             pred_labels = self.model(data)  # one forward pass

    #             loss = self.criterion(pred_labels, labels)
    #             valid_losses.append(loss.item())

    #             accuracy = self.calculate_accuracy(pred_labels, labels)
    #             valid_accuracies.append(accuracy)

    #             current_lr = self.optimizer.param_groups[0]['lr']

    #             batch_iter.set_description(f'Validation: (loss {np.mean(valid_losses):.5f}, Acc {np.mean(valid_accuracies):.2f} LR {current_lr})')  # update progressbar

    #     self.validation_loss.append(np.mean(valid_losses))
    #     self.validation_accuracy.append(np.mean(valid_accuracies))

    #     batch_iter.close()