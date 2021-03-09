import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from Configuration import Config
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


def train_test_data_splitting(dataset, val_split=0.25):
    train_idx, test_idx = train_test_split((list(range(1101))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'test': Subset(dataset, test_idx)}
    return datasets


def train(model, data_place, data_cast, data_action, data_audio):
    model.train()
    output = model(data_place, data_cast, data_action, data_audio)
    return output


class CosineSimilarity(nn.Module):
    def __init__(self, cfg):
        super(CosineSimilarity, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model_settings['sim_channel']
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(self.shot_num//2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num//2]*2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel
        return x


class BNet(nn.Module):
    def __init__(self, cfg):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model_settings['sim_channel']
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = CosineSimilarity(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.view(x.shape[0]*x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class MMSSone(nn.Module):
    def __init__(self, cfg, mode):
        super(MMSSone, self).__init__()
        self.seq_len = cfg.model_settings['seq_len']
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model_settings['lstm_hidden_size']
        if mode == "place":
            self.input_dim = (cfg.model_settings['place_feat_dim'] + cfg.model_settings['sim_channel'])
            self.bnet = BNet(cfg)
        elif mode == "cast":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model_settings['cast_feat_dim'] + cfg.model_settings['sim_channel'])
        elif mode == "act":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model_settings['act_feat_dim'] + cfg.model_settings['sim_channel'])
        elif mode == "aud":
            self.bnet = BNet(cfg)
            self.input_dim = cfg.model_settings['aud_feat_dim']
        else:
            pass
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=cfg.model_settings['bidirectional'])

        self.fc1 = nn.Linear(self.lstm_hidden_size * 2, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        # torch.Size([128, seq_len, 3*channel])
        self.lstm.flatten_parameters()
        out, (_, _) = self.lstm(x, None)
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out.view(-1, 2)
        return out


class MMSS(nn.Module):
    def __init__(self, cfg):
        super(MMSS, self).__init__()
        self.seq_len = cfg.model_settings['seq_len']
        self.mode = cfg.data_settings['mode']
        self.num_layers = 1
        self.lstm_hidden_size = cfg.model_settings['lstm_hidden_size']
        self.ratio = cfg.model_settings['ratio']
        if 'place' in self.mode:
            self.bnet_place = MMSSone(cfg, "place")
        if 'cast' in self.mode:
            self.bnet_cast = MMSSone(cfg, "cast")
        if 'act' in self.mode:
            self.bnet_act = MMSSone(cfg, "act")
        if 'aud' in self.mode:
            self.bnet_aud = MMSSone(cfg, "aud")

    def forward(self, place_features, cast_features, action_features, audio_features):
        out = 0
        if 'place' in self.mode:
            place_bound = self.bnet_place(place_features)
            out += self.ratio[0] * place_bound
        if 'cast' in self.mode:
            cast_bound = self.bnet_cast(cast_features)
            out += self.ratio[1] * cast_bound
        if 'act' in self.mode:
            act_bound = self.bnet_act(action_features)
            out += self.ratio[2] * act_bound
        if 'aud' in self.mode:
            aud_bound = self.bnet_aud(audio_features)
            out += self.ratio[3] * aud_bound
        return out


if __name__ == "__main__":
    with open('./data/tt0052357.pkl', 'rb') as file:
        data = pickle.load(file)
    place_features = torch.randn(128, 10, 4, 2048)
    cast_features = torch.randn(128, 10, 4, 512)
    action_features = torch.randn(128, 10, 4, 512)
    audio_features = torch.randn(128, 10, 4, 512)

    # place_features = data['place'][3]
    # print(place_features)
    # cast_features = data['cast'][3]
    # action_features = data['action'][3]
    # audio_features = data['audio'][3]
    shot_num = 4
    data_settings = dict(mode=['place', 'cast', 'action', 'audio'])
    model_settings = dict(
        name='MMSS',
        sim_channel=512,  # dim of similarity vector
        place_feat_dim=2048,
        cast_feat_dim=512,
        act_feat_dim=512,
        aud_feat_dim=512,
        aud=dict(cos_channel=512),
        seq_len=10,  # even
        bidirectional=True,
        lstm_hidden_size=512,
        ratio=[0.5, 0.2, 0.2, 0.1]
    )
    cfg = Config(data_settings, model_settings, shot_num)
    model = MMSS(cfg)
    output = model(place_features, cast_features, action_features, audio_features)
    print(output.data)

    with open('./data/tt0052357.pkl', 'rb') as file:
        data = pickle.load(file)

    place_features = data['place']
    cast_features = data['cast']
    action_features = data['action']
    audio_features = data['audio']
    data_place = train_test_data_splitting(place_features)
    data_cast = train_test_data_splitting(cast_features)
    data_action = train_test_data_splitting(action_features)
    data_audio = train_test_data_splitting(audio_features)
    print(data_place)

    scheduler = dict(name='MultiStepLR', setting=dict(milestones=[15]))
    criterion = nn.CrossEntropyLoss(torch.Tensor([0.5, 5]))

    trainFlag = 1
    if trainFlag:
        max_ap = -1
        epochs = 30
        for epoch in range(1, epochs + 1):
            print("Woof")
            print(train(model, data_place['train'], data_cast['train'], data_action['train'], data_audio['train']))
