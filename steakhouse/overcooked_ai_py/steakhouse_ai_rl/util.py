# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from steakhouse_ai_py.mdp.steakhouse_mdp import SteakhouseGridworld, dishname2ingradient, ingradient2dishname
from steakhouse_ai_py.mdp.steakhouse_env import SteakhouseEnv

import copy
import logging
from hydra.core.utils import configure_log
from omegaconf import read_write



NO_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 0,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
    "COOKING_STEAK_REW": 0,
}

BASE_REW_SHAPING_PARAMS = {
    "ONION_PICKUP_REWARD": 1,
    "MEAT_PICKUP_REWARD": 1,
    "DIRTY_PLATE_PICKUP_REWARD": 1,
    "CHICKEN_PICKUP_REWARD": 0,
    # drop raw ingredient actions
    "DROP_DIRTY_PLATE": 2,
    "PLACEMENT_IN_POT_REW": 2,
    "PLACEMENT_IN_GRILL_REW": 2,
    "PLACEMENT_ON_BOARD_REW": 2,
    # perform actions on raw ingredients
    "RINSE_DIRTY_PLATE": 1,
    "CHOPPING_ONION_REW": 1,
    # "COMPLETED_RINSE": 4,
    # pick up processed ingredients
    "CLEAN_PLATE_PICKUP_REWARD":4,
    # pick up processed ingredients with clean plate
    "BOILED_CHICKEN_PICKUP_REWARD": 8,
    "STEAK_PICKUP_REWARD": 8,
    "GARNISH_STEAK_REWARD": 10,
}

def init_env(args):
    layout_name = args.env.layout
    layout_file_name = layout_name + ".layout"

    # Copy layout to Overcooked AI code base
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
    path_from = os.path.join(base_folder, "src/data", "layout", layout_file_name)
    path_to = os.path.join(base_folder, "overcooked_ai", "src",
                            "overcooked_ai_py", "data", "layouts", layout_file_name)
    shutil.copy(path_from, path_to)
    world_mdp = SteakhouseGridworld.from_layout_name(layout_name)

    if 'rand_start' in args:
        empty_cell = world_mdp.terrain_pos_dict[' ']
        p1, p2 = random.choices(np.arange(0, len(empty_cell)), k=2)
        world_mdp.start_player_positions = [empty_cell[p1], empty_cell[p2]]
        
    env = SteakhouseEnv.from_mdp(world_mdp, horizon=args.env.total_time)

    return env

def make_env(args):
    def thunk():
        return init_env(args)
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NNAgent(nn.Module):
    def __init__(self, envs=None, obs_size=None, action_size=None, hidden_size=64):
        super().__init__()

        self.obs_size = obs_size if obs_size else np.array(envs.single_observation_space.shape).prod()
        self.action_size = action_size if action_size else len(self.ml_action_list)
        self.hidden_size = hidden_size

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_size, self.action_size), std=0.01),
        )

        
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        probs.log_prob(action)
        probs.entropy()
        self.critic(x)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def setup_logging(hydra_cfg, dask_worker = None):
    """Sets up the worker logger."""
    logdir = hydra_cfg.dask.output_dir
    os.makedirs(f"{logdir}/worker_log/", exist_ok=True)

    # Get the default config and modify the format and the output file
    log_config = copy.deepcopy(hydra_cfg.dask.job_logging)

    log_file = f"{logdir}/worker_log/{dask_worker.name}.log"
    with read_write(log_config):
        log_config["handlers"]["file"]["filename"] = log_file

    configure_log(log_config)

def worker_log(msg: str, level: int = logging.INFO):
    """Logs a message on the worker.

    Intended to be used to run a logging message on all workers with
    dask.distributed.Client.run, e.g.

        client.run(worker_log, "this is a message")
    """
    logger = logging.getLogger(__name__)
    logger.log(level, msg)


class LSTM_Agent(nn.Module):
    def __init__(self, obs_size=None, action_size=None, lstm_size=128, num_layers=1):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(self.obs_size, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, self.lstm_size, num_layers=self.num_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(self.lstm_size, self.action_size), std=0.01)
        self.critic = layer_init(nn.Linear(self.lstm_size, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state

    #MISHA NEW CHANGE FOR LSTM TRAINING
    def forward(self, x, lstm_state=None, done=None):
        """For BC: returns logits (before softmax) over actions."""
        hidden, _ = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        return logits, None  # match expected return shape (logits, hidden)

    #END MISHA NEW CHANGE

class Utter_Agent(nn.Module):
    def __init__(self, obs_size=None, action_size=None, hist_utter_size=10, lstm_size=128, lstm_layers_n=1):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.lstm_size = lstm_size
        self.hist_utter_size = hist_utter_size
        self.lstm_layers_n = lstm_layers_n
        self.network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(self.obs_size, 2048)),
            nn.LayerNorm(2048),
            nn.ReLU(),
            # layer_init(nn.Linear(2048, 1024)),
            # nn.LayerNorm(1024),
            # nn.ReLU(),
            layer_init(nn.Linear(2048, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, self.lstm_size, num_layers=self.lstm_layers_n)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(self.lstm_size + self.hist_utter_size, self.action_size), std=0.01)
        self.critic = layer_init(nn.Linear(self.lstm_size + self.hist_utter_size, 1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, hist_utter, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        hidden = torch.cat((hidden, hist_utter), 1)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, hist_utter, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        hidden = torch.cat((hidden, hist_utter), 1)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
