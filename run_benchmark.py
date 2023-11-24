import os
import shutil
from pathlib import Path
import yaml

import acme
from acme import wrappers
import acme.utils.loggers as log_utils
import dm_env


from domain.sde import *
from domain.asset.base import *
from domain.asset.portfolio import Portfolio
from env.trade_env import DREnv
from config.config_loader import ConfigLoader
from agent.benchmark_agent import DeltaHedgeAgent, GammaHedgeAgent
from analysis.gen_stats import generate_stat

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('benchmark_name', 'DeltaHedging', 'Benchmark name - DeltaHedging or GammaHedging (Default DeltaHedging)')
flags.DEFINE_integer('eval_sim', 10000, 'Number of evaluation episodes (Default 10000)')
flags.DEFINE_string('env_config', '', 'Environment config (Default None)')
flags.DEFINE_integer('evaluator_seed', 4321, 'Evaluation Seed (Default 4321)')
flags.DEFINE_string('logger_prefix', 'logs/', 'Prefix folder for logger (Default None)')


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


def make_loggers(work_folder):
    return dict(
        train_loop=make_logger(work_folder, 'train_loop', terminal=True),
        eval_loop=make_logger(work_folder, 'eval_loop', terminal=True),
        learner=make_logger(work_folder, 'learner')
    )


def make_environment(label, seed=1234) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    with open(FLAGS.env_config, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)
    config_loader = ConfigLoader(config_data)
    config_loader.load_objects()
    environment: DREnv = config_loader[label] if label in config_loader.objects else config_loader['env']
    environment.seed(seed)
    environment.logger = make_logger(FLAGS.logger_prefix, label)
    environment = wrappers.GymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def main(argv):
    work_folder = FLAGS.logger_prefix
    if os.path.exists(work_folder):
        shutil.rmtree(work_folder)
    # Create an environment, grab the spec, and use it to create networks.
    loggers = make_loggers(work_folder=work_folder)
    # Construct the agent.
    if FLAGS.benchmark_name == 'DeltaHedging':
        agent = DeltaHedgeAgent() 
    elif FLAGS.benchmark_name == 'GammaHedging':
        agent = GammaHedgeAgent()
    else:
        raise NotImplementedError(f'Benchmark {FLAGS.benchmark_name} not implemented.')

    eval_env = make_environment('eval_env', seed=FLAGS.evaluator_seed)
    eval_loop = acme.EnvironmentLoop(eval_env, agent, label='eval_loop', logger=loggers['eval_loop'])
    eval_loop.run(num_episodes=FLAGS.eval_sim)
    print(generate_stat(f'{work_folder}/logs/eval_env/logs.csv',
                        [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]))

    Path(f'{work_folder}/ok').touch()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('app.run(main)', 'output.prof')
    app.run(main)
