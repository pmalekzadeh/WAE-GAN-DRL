import os
import shutil
from pathlib import Path
import yaml
from typing import Mapping, Sequence

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
from config.config_loader import ConfigLoader
from agent.agent_distributed import DistributedD4PG as D4PG
import agent.distributional as ad
from analysis.gen_stats import generate_stat
from agent.demonstrations import DemonstrationRecorder

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('env_config', '', 'Environment config (Default None)')
flags.DEFINE_integer('max_actor_episodes', 50000, 'Maximum number of actor episodes for training (Default 50000)')
flags.DEFINE_integer('num_actors', 5, 'Number of actors (Default 5)')
flags.DEFINE_list('actor_seeds', ['2345', '3456', '4567', '5678'], 'Actor seeds (Default [2345, 3456, 4567, 5678])')
flags.DEFINE_integer('evaluator_seed', 4321, 'Evaluation Seed (Default 4321)')
flags.DEFINE_integer('n_step', 5, 'DRL TD Nstep (Default 5)')
flags.DEFINE_string('critic', 'qr', 'critic distribution type - c51 or qr (Default qr, we use quantile regression to esitmate the output of the critic for all objective functions)')
flags.DEFINE_string('obj_func', 'var', 'Objective function select from meanstd, var or cvar (Default var)')
flags.DEFINE_float('std_coef', 1.645, 'Std coefficient when obj_func=meanstd. (Default 1.645)')
flags.DEFINE_float('threshold', 0.99, 'Objective function threshold. (Default 0.99)')
flags.DEFINE_string('logger_prefix', 'logs/', 'Prefix folder for logger (Default None)')
flags.DEFINE_boolean('per', False, 'Use PER for Replay sampling (Default False)')
flags.DEFINE_float('importance_sampling_exponent', 0.2, 'importance sampling exponent for updating importance weight for PER (Default 0.2)')
flags.DEFINE_float('priority_exponent', 0.6, 'priority exponent for the Prioritized replay table (Default 0.6)')
flags.DEFINE_float('lr', 1e-4, 'Learning rate for optimizer (Default 1e-4)')
flags.DEFINE_integer('batch_size', 256, 'Batch size to train the Network (Default 256)')
flags.DEFINE_integer('buffer_steps', 0, 'Buffer Steps in Transaction Cost Case (Default 0)')
flags.DEFINE_string('demo_path', '', 'Demo Path (Default None)')
flags.DEFINE_integer('demo_step', 1000, 'Demo Steps (Default 1000)')


def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(work_folder,
                            label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))

    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger


def make_environment(label='', seed=1234) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    with open(FLAGS.env_config, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    config_loader = ConfigLoader(config_data)
    config_loader.load_objects()
    if 'actor' in label:
        env_type = 'train_env'
    elif 'evaluator' in label:
        env_type = 'eval_env'
    else:
        env_type = 'train_env'

    environment: DREnv = config_loader[env_type] if env_type in config_loader.objects else config_loader['env']
    environment.seed(seed)
    if label != '':
        environment.logger = make_logger(FLAGS.logger_prefix, label, terminal=False)
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
    work_folder = FLAGS.logger_prefix
    if os.path.exists(work_folder):
        shutil.rmtree(work_folder)
    # Create an environment, grab the spec, and use it to create networks.
    if FLAGS.critic == 'c51':
        agent_networks_factory = make_networks
    elif FLAGS.critic == 'qr':
        agent_networks_factory = make_quantile_networks
    elif FLAGS.critic == 'iqn':
        assert FLAGS.obj_func == 'cvar', 'IQN only support CVaR objective.'
        agent_networks_factory = lambda a_spec: make_iqn_networks(action_spec=a_spec, cvar_th=FLAGS.threshold)
    # Construct the agent.
    agent = D4PG(
        environment_factory=make_environment,
        network_factory=agent_networks_factory,
        num_actors=FLAGS.num_actors,
        actor_seeds=[int(s) for s in FLAGS.actor_seeds],
        evaluator_seed=FLAGS.evaluator_seed,
        obj_func=FLAGS.obj_func,
        threshold=FLAGS.threshold,
        critic_loss_type=FLAGS.critic,
        n_step=FLAGS.n_step,
        sigma=0.3,  # pytype: disable=wrong-arg-types
        checkpoint=True,
        batch_size=FLAGS.batch_size,
        per=FLAGS.per,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=FLAGS.importance_sampling_exponent,
        demonstration_step=FLAGS.demo_step,
        demonstration_dataset=None if FLAGS.demo_path == ''
            else DemonstrationRecorder().load(FLAGS.demo_path).make_tf_dataset(),
        discount=1.0,
        policy_optimizer=snt.optimizers.Adam(FLAGS.lr),
        critic_optimizer=snt.optimizers.Adam(FLAGS.lr),
        max_actor_episodes=FLAGS.max_actor_episodes,
        log_folder=work_folder,
        checkpoint_folder=work_folder,
    )
    program = agent.build()
    
    lp.launch(program, launch_type='local_mt')
    Path(f'{work_folder}/ok').touch()


if __name__ == '__main__':
    app.run(main)
