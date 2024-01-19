import os
import torch
from torch import nn
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
import intel_extension_for_pytorch as ipex
import numpy as np
import wavio
import wave
import shutil
from pyache import Pyache

from vectorization import *
from precise.params import pr, Vectorizer

LOSS_BIAS = 0.9
SAMPLE_RATE = 16000
BATCH_SIZE = 8
LR_SCHEDULER_STEP_SIZE = 100
LR_SCHEDULER_GAMMA = 0.9
LR_SCHEDULER_INIT_LR = 5e-3
TENSORBOARD_SUMMARY_DIR = "./tensorboard_summary"
MAX_EPOCH = 100
VAL_EACH_ITER = 20
class InvalidAudio(ValueError):
    """Thrown the audio isn't in the expected format"""
    pass


class dataloader(dataset.Dataset):
    def __init__(self, path, train=True):
        super(dataloader, self).__init__()
        if train:
            self.wake_word_path = os.path.join(path, "wake-word")
            self.not_wake_word_path = os.path.join(path, "not-wake-word")
        else:
            self.wake_word_path = os.path.join(path, "test", "wake-word")
            self.not_wake_word_path = os.path.join(path, "test", "not-wake-word")

        if not os.path.exists(self.wake_word_path):
            raise RuntimeError("could not find {}".format(self.wake_word_path))
        if not os.path.exists(self.not_wake_word_path):
            raise RuntimeError("could not find {}".format(self.not_wake_word_path))

        self.data_list = []
        self.positive_label_number = 0
        self.negative_label_number = 0

        for audio_file in os.listdir(self.wake_word_path):
            if audio_file.endswith(".wav"):
                self.data_list.append((os.path.join(self.wake_word_path, audio_file), 1))
                self.positive_label_number += 1

        if self.positive_label_number < 8:
            raise RuntimeError("positive_label_number must great than 12, but given is {}".format(self.positive_label_number))

        for audio_file in os.listdir(self.not_wake_word_path):
            if audio_file.endswith(".wav"):
                self.data_list.append((os.path.join(self.not_wake_word_path, audio_file), 0))
                self.negative_label_number += 1

        if self.negative_label_number < 4:
            raise RuntimeError("negative_label_number must great than 4, but given is {}".format(self.negative_label_number))

        pass

    def __getitem__(self, index):
        audio_filename, label = self.data_list[index]

        vectorizer = (vectorize_delta if pr.use_delta else vectorize)
        cache = Pyache('.cache', lambda x: vectorizer(load_audio(x)), pr.vectorization_md5_hash())
        audio_np = cache.load([audio_filename])

        label_np = np.array([label], dtype=np.float32)

        audio_tensor = torch.from_numpy(audio_np).squeeze()
        label_tensor = torch.from_numpy(label_np)
        return audio_tensor, label_tensor

    def __len__(self):
        return len(self.data_list)


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def weighted_log_loss(yt, yp):
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    pos_loss = -(0 + yt) * torch.function.log(0 + yp + torch.function.epsilon())
    neg_loss = -(1 - yt) * torch.function.log(1 - yp + torch.function.epsilon())
    return LOSS_BIAS * torch.function.mean(neg_loss) + (1. - LOSS_BIAS) * torch.function.mean(pos_loss)


def train_and_validate(dataset_path, training_device='cpu'):
    train_dataset = dataloader(dataset_path, train=True)
    val_dataset = dataloader(dataset_path, train=False)

    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = torch.utils.data.dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_mini_batch_number = len(train_loader)
    val_mini_batch_number = len(val_loader)

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, device=training_device, drop_prob=0.2)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_SCHEDULER_INIT_LR)
    step_schedule = torch.optim.lr_scheduler.StepLR(step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA, optimizer=optimizer)

    model.train()
    model.to(training_device)
    print(model)

    if os.path.exists(TENSORBOARD_SUMMARY_DIR):
        shutil.rmtree(TENSORBOARD_SUMMARY_DIR)
    os.makedirs(TENSORBOARD_SUMMARY_DIR)

    writer = SummaryWriter(log_dir=TENSORBOARD_SUMMARY_DIR)

    dummy_audio = torch.rand(BATCH_SIZE, pr.n_features, pr.feature_size).to(training_device)
    writer.add_graph(model=model, input_to_model=[dummy_audio])

    iter_total = 1
    iter_val_total = 0
    times_val = 0

    for epoch in range(MAX_EPOCH):
        for iter, (audio_tensor, label_tensor) in enumerate(train_loader):
            audio_tensor = audio_tensor.to(training_device)
            label_tensor = label_tensor.to(training_device)

            logits = model(audio_tensor)
            loss = weighted_log_loss(logits, label_tensor)
            print("epoch {}/{}, iter {}/{}, loss = {}".format(epoch + 1, MAX_EPOCH, iter + 1, train_mini_batch_number, loss))
            writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=iter_total - 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_schedule.step()
            writer.add_scalar(tag="train/lr", scalar_value=step_schedule.get_last_lr()[0], global_step=iter_total - 1)
            if iter_total % VAL_EACH_ITER == 0:
                print("its time to val")
                correct_count = 0
                camera_correct_count = 0
                with torch.no_grad():
                    model.eval()
                    for val_iter, (val_image, val_label, feature) in enumerate(val_loader):
                        audio_tensor_val = audio_tensor.to(training_device)
                        label_tensor_val = label_tensor.to(training_device)
                        val_logits = model(audio_tensor_val)

                        val_loss = weighted_log_loss(val_logits, val_label)
                        val_result = torch.argmax(val_logits, dim=1, keepdim=False)
                        correct_count += torch.sum(val_label == val_result)

                        print("val, iter {}/{}, loss = {}".format(val_iter + 1, val_mini_batch_number, val_loss))
                        writer.add_scalar(tag="val/loss", scalar_value=val_loss, global_step=iter_val_total)
                        # writer.add_scalar(tag="val/loss_a", scalar_value=val_loss_a, global_step=iter_val_total)
                        # writer.add_scalar(tag="val/loss_b", scalar_value=val_loss_b, global_step=iter_val_total)
                        iter_val_total += 1
                    acc = correct_count / (val_mini_batch_number * BATCH_SIZE)
                    print("val acc = {}".format(acc))
                    writer.add_scalar(tag="val/acc", scalar_value=acc, global_step=times_val)

                    times_val += 1
                    model.train()

            iter_total += 1

    pass


if __name__ == '__main__':
    dummy_dataloader = dataloader("/home/anna/WorkSpace/celadon/demo-src/precise/training/data")
    dummy_loader = torch.utils.data.dataloader.DataLoader(
        dataset=dummy_dataloader,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    for audio_tensor, label_tensor in dummy_loader:
        print(audio_tensor.shape)
        print(label_tensor.shape)
