import os
import torch
from torch import nn
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
# import intel_extension_for_pytorch as ipex
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
MODEL_NAME = "./torch_model.pth"
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
        audio_np = cache.load([audio_filename]).astype(np.float32)

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

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob, bias=False)
        self.fc = nn.Linear(hidden_dim * hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        gru_out_y, gru_out_h = self.gru(x, h)  # [batch_size, 29, 29], [20, batch_size, 29]
        gru_out_y_vec = gru_out_y.reshape(gru_out_y.shape[0], -1)
        # out = self.fc(self.sigmoid1(out[:, -1]))
        out = self.fc(gru_out_y_vec)
        out = self.sigmoid(out)
        return out, gru_out_h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


def weighted_log_loss(yt: torch.Tensor, yp: torch.Tensor):
    """
    Binary cross entropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    cross entropy = Mean[- (yt*log(yp) + (1-yt)*log(1-yp))]
    when yt = 0, cross entropy = Mean[- log(1- yp)], false positive
    when yt = 1, cross entropy = Mean[- log(yp)], false negative
    """

    pos_loss = -(0 + yt) * torch.log(0 + yp + 1e-7)
    neg_loss = -(1 - yt) * torch.log(1 - yp + 1e-7)

    return LOSS_BIAS * torch.mean(neg_loss) + (1. - LOSS_BIAS) * torch.mean(pos_loss)

    # pos_loss = torch.mean(-(0 + yt) * torch.log(0 + yp + 1e-7))
    # neg_loss = torch.mean(-(1 - yt) * torch.log(1 - yp + 1e-7))
    # return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss

    # cross_entropy_core = - (yt * torch.log(yp) + (1 - yt) * torch.log(1 - yp+ 1e-7))
    # true_mask = yt * LOSS_BIAS * cross_entropy_core
    # false_mask = (1 - yt) * (1. - LOSS_BIAS) * cross_entropy_core
    # return torch.mean(cross_entropy_core)



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

    print("pr.n_features = {}".format(pr.n_features))
    print("pr.feature_size = {}".format(pr.feature_size))

    model = GRUNet(input_dim=pr.feature_size, hidden_dim=pr.n_features, output_dim=1, n_layers=20, device=training_device, drop_prob=0.2)
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
    dummy_h = model.init_hidden(dummy_audio.shape[0])
    writer.add_graph(model=model, input_to_model=[dummy_audio, dummy_h])

    iter_total = 1
    iter_val_total = 0
    times_val = 0

    for epoch in range(MAX_EPOCH):
        for iter, (audio_tensor, label_tensor) in enumerate(train_loader):
            audio_tensor = audio_tensor.to(training_device)  # [batch_size, 29, 13]
            label_tensor = label_tensor.to(training_device)  # [batch_size, 1]
            h = model.init_hidden(audio_tensor.shape[0])  # [20, batch_size, 29]
            output_y, output_h = model(audio_tensor, h)  # [batch_size, 1], [20, batch_size, 29]
            loss = weighted_log_loss(label_tensor, output_y)
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
                    for val_iter, (val_audio_tensor, val_label_tensor) in enumerate(val_loader):
                        audio_tensor_val = val_audio_tensor.to(training_device)
                        label_tensor_val = val_label_tensor.to(training_device)
                        h = model.init_hidden(audio_tensor_val.shape[0])
                        val_y, val_h = model(audio_tensor_val, h)

                        val_loss = weighted_log_loss(label_tensor_val, val_y)
                        val_result = (val_y > 0.5).float()
                        correct_count += torch.sum(label_tensor_val == val_result)

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
    writer.close()
    torch.save(model, MODEL_NAME)

if __name__ == '__main__':
    # dummy_dataloader = dataloader("/home/anna/WorkSpace/celadon/demo-src/precise/training/data")
    # dummy_loader = torch.utils.data.dataloader.DataLoader(
    #     dataset=dummy_dataloader,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True
    # )
    # for audio_tensor, label_tensor in dummy_loader:
    #     print(audio_tensor.shape)
    #     print(label_tensor.shape)

    train_and_validate(dataset_path="/home/anna/WorkSpace/celadon/demo-src/precise/training/data",
                       training_device='cpu')

