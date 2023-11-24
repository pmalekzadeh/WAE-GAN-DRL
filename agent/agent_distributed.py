# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the D4PG agent class."""

import copy
from typing import Callable, Dict, List, Optional

import acme
from acme import specs
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
import acme.utils.loggers as log_utils
from acme.utils import lp_utils
from acme.adders import reverb as reverb_adders
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf

from agent.agent import D4PGConfig, D4PGBuilder, D4PGNetworks

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


class DistributedD4PG:
  """Program definition for D4PG."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
      num_actors: int = 1,
      actor_seeds: List[int] = [1234],
      evaluator_seed: int = 4321,
      num_caches: int = 0,
      obj_func='var',
      critic_loss_type='c51',
      threshold=0.95,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      batch_size: int = 256,
      prefetch_size: int = 4,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: Optional[float] = 32.0,
      n_step: int = 5,
      sigma: float = 0.3,
      clipping: bool = True,
      per: bool = False,
      priority_exponent: float = 0.6,
      importance_sampling_exponent: float = 0.2,
      replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
      demonstration_step: int = 1_000,
      demonstration_dataset: Optional[tf.data.Dataset] = None,  
      discount: float = 0.99,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      target_update_period: int = 100,
      max_actor_episodes: Optional[int] = None,        # Max actor steps to terminate.
      log_folder: Optional[str] = None,
      log_every: float = 10.0,
      checkpoint: bool = True,
      checkpoint_folder: str = '~/acme'
  ):

    if not environment_spec:
      environment_spec = specs.make_environment_spec(environment_factory(seed=1234))

    # TODO(mwhoffman): Make network_factory directly return the struct.
    # TODO(mwhoffman): Make the factory take the entire spec.
    def wrapped_network_factory(action_spec):
      networks_dict = network_factory(action_spec)
      networks = D4PGNetworks(
          policy_network=networks_dict.get('policy'),
          critic_network=networks_dict.get('critic'),
          observation_network=networks_dict.get('observation', tf.identity))
      return networks
    self._log_folder = log_folder
    self._environment_factory = environment_factory
    self._network_factory = wrapped_network_factory
    self._environment_spec = environment_spec
    self._sigma = sigma
    self._num_actors = num_actors
    self._actor_seeds = actor_seeds
    assert len(actor_seeds) == num_actors, "actor_seeds must be the same length as num_actors"
    self._evaluator_seed = evaluator_seed
    self._num_caches = num_caches
    self._max_actor_episodes = max_actor_episodes
    self._log_every = log_every
    self._demonstration_dataset = demonstration_dataset
    self._checkpoint = checkpoint
    self._checkpoint_folder = checkpoint_folder
    self._replay_table_name = replay_table_name

    self._builder = D4PGBuilder(
        # TODO(mwhoffman): pass the config dataclass in directly.
        # TODO(mwhoffman): use the limiter rather than the workaround below.
        D4PGConfig(
            obj_func=obj_func,
            critic_loss_type=critic_loss_type,
            threshold=threshold,
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            n_step=n_step,
            sigma=sigma,
            demo_step=demonstration_step,
            clipping=clipping,
            per=per,
            priority_exponent=priority_exponent,
            importance_sampling_exponent=importance_sampling_exponent,
            replay_table_name=replay_table_name,
        ))

  def replay(self):
    """The replay storage."""
    return self._builder.make_replay_tables(self._environment_spec)

  def counter(self):
    return tf2_savers.CheckpointingRunner(counting.Counter(),
                                          time_delta_minutes=1,
                                          subdirectory='counter')

  def coordinator(self, counter: counting.Counter):
    return lp_utils.StepsLimiter(counter, self._max_actor_episodes, steps_key='actor_episodes')

  def learner(
      self,
      replay: reverb.Client,
      counter: counting.Counter,
  ):
    """The Learning part of the agent."""

    # Create the networks to optimize (online) and target networks.
    online_networks = self._network_factory(self._environment_spec.actions)
    target_networks = copy.deepcopy(online_networks)

    networks = (online_networks, target_networks)
    # Initialize the networks.
    online_networks.init(self._environment_spec)
    target_networks.init(self._environment_spec)
    
    dataset, demo_dataset = self._builder.make_dataset_iterator(replay, self._demonstration_dataset)
    counter = counting.Counter(counter, 'learner')
    
    return self._builder.make_learner(
        networks=networks,
        dataset=dataset,
        demo_dataset=demo_dataset,
        counter=counter,
        logger=make_logger(self._log_folder, 'learner'),
        checkpoint=self._checkpoint,
        checkpoint_folder=self._checkpoint_folder,
        replay_client=replay,
    )

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      actor_id: int = 0,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    # Create the behavior policy.
    networks = self._network_factory(self._environment_spec.actions)
    networks.init(self._environment_spec)
      
    policy_network = networks.make_policy(
        environment_spec=self._environment_spec,
        sigma=self._sigma,
    )

    # Create the agent.
    actor = self._builder.make_actor(
        policy_network=policy_network,
        adder=self._builder.make_adder(replay),
        variable_source=variable_source,
    )

    # Create the environment.
    environment = self._environment_factory(f'actor{actor_id}_env', self._actor_seeds[actor_id])

    # Create logger and counter; actors will not spam bigtable.
    counter = counting.Counter(counter, 'actor')
    logger = make_logger(self._log_folder, f'actor{actor_id}_loop')
    loggers.make_default_logger(
        'actor',
        save_data=False,
        time_delta=self._log_every,
        steps_key='actor_steps')
    # TODO (mwhoffman): Continue here add specific loggers
    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      seed: int = 4321,
  ):
    """The evaluation process."""

    # Create the behavior policy.
    networks = self._network_factory(self._environment_spec.actions)
    networks.init(self._environment_spec)
    policy_network = networks.make_policy(self._environment_spec)

    # Create the agent.
    actor = self._builder.make_actor(
        policy_network=policy_network,
        variable_source=variable_source,
    )

    # Make the environment.
    environment = self._environment_factory('evaluator_env', seed)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = make_logger(self._log_folder, 'evaluator_loop', terminal=True)

    # Create the run loop and return it.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='d4pg'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    if self._max_actor_episodes:
      with program.group('coordinator'):
        _ = program.add_node(lp.CourierNode(self.coordinator, counter))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, learner, counter, self._evaluator_seed))

    if not self._num_caches:
      # Use our learner as a single variable source.
      sources = [learner]
    else:
      with program.group('cacher'):
        # Create a set of learner caches.
        sources = []
        for _ in range(self._num_caches):
          cacher = program.add_node(
              lp.CacherNode(
                  learner, refresh_interval_ms=2000, stale_after_ms=4000))
          sources.append(cacher)

    with program.group('actor'):
      # Add actors which pull round-robin from our variable sources.
      for actor_id in range(self._num_actors):
        source = sources[actor_id % len(sources)]
        program.add_node(lp.CourierNode(self.actor, replay, source, counter, actor_id))

    return program