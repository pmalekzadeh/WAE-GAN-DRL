import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence
from functools import partial

import tensorflow as tf
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
import acme.utils.loggers as log_utils
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf.savers import make_snapshot
import dm_env
import numpy as np
import sonnet as snt
import launchpad as lp


from domain.sde import *
from domain.asset.base import *
from domain.asset.portfolio import Portfolio
from env.trade_env import DREnv
from config.config_loader import ConfigLoader, save_args_to_file
from agent.agent_distributed import DistributedD4PG as D4PG
import agent.distributional as ad
from analysis.gen_stats import generate_stat
from agent.demonstrations import DemonstrationRecorder

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--env_config', type=str, default='', help='Environment config')
parser.add_argument('--max_actor_episodes', type=int, default=50000, help='Maximum number of actor episodes for training')
parser.add_argument('--num_actors', type=int, default=5, help='Number of actors')
parser.add_argument('--actor_seeds', type=lambda s: [int(item) for item in s.split()], default=[2345, 3456, 4567, 5678, 6789], help='Actor seeds')
parser.add_argument('--evaluator_seed', type=int, default=4321, help='Evaluation Seed')
parser.add_argument('--n_step', type=int, default=5, help='DRL TD Nstep')
parser.add_argument('--critic', type=str, default='qr', help='critic distribution type - c51 or qr')
parser.add_argument('--obj_func', type=str, default='var', help='Objective function select from meanstd, var or cvar')
parser.add_argument('--std_coef', type=float, default=1.645, help='Std coefficient when obj_func=meanstd.')
parser.add_argument('--threshold', type=float, default=0.99, help='Objective function threshold.')
parser.add_argument('--logger_prefix', type=str, default='logs/', help='Prefix folder for logger')
parser.add_argument('--per', action='store_true', help='Use PER for Replay sampling')
parser.add_argument('--importance_sampling_exponent', type=float, default=0.2, help='importance sampling exponent for updating importance weight for PER')
parser.add_argument('--priority_exponent', type=float, default=0.6, help='priority exponent for the Prioritized replay table')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size to train the Network')
parser.add_argument('--buffer_steps', type=int, default=0, help='Buffer Steps in Transaction Cost Case')
parser.add_argument('--demo_path', type=str, default='', help='Demo Path')
parser.add_argument('--demo_step', type=int, default=1000, help='Demo Steps')



def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(work_folder,
                            label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label, print_fn=print))

    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger


def make_environment(env_config_file, env_cmd_args, logger_prefix, label='', seed=1234) -> dm_env.Environment:
    config_loader = ConfigLoader(config_file=env_config_file, cmd_args=env_cmd_args)
    config_loader.load_objects()
    config_loader.save_config(os.path.join(logger_prefix, 'env.yaml'))
    if 'actor' in label:
        env_type = 'train_env'
    elif 'evaluator' in label:
        env_type = 'eval_env'
    else:
        env_type = 'train_env'

    environment: DREnv = config_loader[env_type] if env_type in config_loader.objects else config_loader['env']
    environment.seed(seed)
    environment.logger = make_logger(logger_prefix, label, terminal=False)
    environment = wrappers.GymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment

# The default settings in this network factory will work well for the
# TradingENV task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.RiskDiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


def make_quantile_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.QuantileDiscreteValuedHead(
            quantiles=quantiles, prob_type=ad.QuantileDistProbType.MID),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


def make_iqn_networks(
    action_spec: specs.BoundedArray,
    cvar_th: float,
    n_cos=64, n_tau=8, n_k=32,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    # Create the critic network.
    critic_network = ad.IQNCritic(
        cvar_th, n_cos, n_tau, n_k, critic_layer_sizes, quantiles, ad.QuantileDistProbType.MID)

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


def save_policy(policy_network, checkpoint_folder):
    snapshot = make_snapshot(policy_network)
    tf.saved_model.save(snapshot, checkpoint_folder+'/policy')


def load_policy(policy_network, checkpoint_folder):
    trainable_variables_snapshot = {}
    load_net = tf.saved_model.load(checkpoint_folder+'/policy')
    for var in load_net.trainable_variables:
        trainable_variables_snapshot['/'.join(
            var.name.split('/')[1:])] = var.numpy()
    for var in policy_network.trainable_variables:
        var_name_wo_name_scope = '/'.join(var.name.split('/')[1:])
        var.assign(
            trainable_variables_snapshot[var_name_wo_name_scope])


def main(argv):
    args, env_cmd_args = parser.parse_known_args(argv)
    work_folder = args.logger_prefix
    if os.path.exists(os.path.join(work_folder, 'logs')):
        shutil.rmtree(os.path.join(work_folder, 'logs'))
    os.makedirs(work_folder, exist_ok=True)
    save_args_to_file(args, os.path.join(work_folder, 'agent.cfg'))
    # Create an environment, grab the spec, and use it to create networks.
    if args.critic == 'c51':
        agent_networks_factory = make_networks
    elif args.critic == 'qr':
        agent_networks_factory = make_quantile_networks
    elif args.critic == 'iqn':
        assert args.obj_func == 'cvar', 'IQN only support CVaR objective.'
        agent_networks_factory = lambda a_spec: make_iqn_networks(action_spec=a_spec, cvar_th=args.threshold)
    # Construct the agent.
    agent = D4PG(
        environment_factory=partial(make_environment, env_config_file=args.env_config, 
                                    env_cmd_args=env_cmd_args, logger_prefix=work_folder),
        network_factory=agent_networks_factory,
        num_actors=args.num_actors,
        actor_seeds=args.actor_seeds,
        evaluator_seed=args.evaluator_seed,
        obj_func=args.obj_func,
        threshold=args.threshold,
        critic_loss_type=args.critic,
        n_step=args.n_step,
        discount=1.0,
        sigma=0.3,  # pytype: disable=wrong-arg-types
        checkpoint=True,
        batch_size=args.batch_size,
        per=args.per,
        priority_exponent=args.priority_exponent,
        importance_sampling_exponent=args.importance_sampling_exponent,
        demonstration_step=args.demo_step,
        demonstration_dataset=None if args.demo_path == ''
            else DemonstrationRecorder().load(args.demo_path).make_tf_dataset(),
        policy_optimizer=snt.optimizers.Adam(args.lr),
        critic_optimizer=snt.optimizers.Adam(args.lr),
        max_actor_episodes=args.max_actor_episodes,
        log_folder=work_folder,
        checkpoint_folder=work_folder,
    )
    program = agent.build()
    
    lp.launch(program, launch_type='local_mt')
    # print(generate_stat(f'{work_folder}/logs/eval_env/logs.csv',
    #                    [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]))
    # Path(f'{work_folder}/ok').touch()


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])

