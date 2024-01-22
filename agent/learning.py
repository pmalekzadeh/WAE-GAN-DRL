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

"""D4PG learner implementation."""
import random
import time
from typing import Dict, Iterator, List, Optional, Union

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks as acme_nets
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.adders import reverb as reverb_adders
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree

from agent.distributional import quantile_regression, huber
import tensorflow_probability as tfp
tfd = tfp.distributions


class GANLearner(acme.Learner):
    """D4PG learner.
    This is the learning component of a D4PG agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        gen_network: snt.Module ,
        disc_network: snt.Module,
        encoder_loc_network: snt.Module,
        encoder_scale_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        target_gen_network: snt.Module,
        target_disc_network: snt.Module,
        target_encoder_loc_network: snt.Module,
        target_encoder_scale_network: snt.Module,
        discount: float,
        target_update_period: int,
        dataset_iterator: Iterator[reverb.ReplaySample],
        demo_dataset_iterator: Optional[Iterator[reverb.ReplaySample]] = None,
        demo_step: Optional[int] = 1_000,
        obj_func='var',
        critic_loss_type='c51',
        threshold=0.95,
        decay_factor=.6,
        z_dim =2,
        observation_network: types.TensorTransformation = lambda x: x,
        target_observation_network: types.TensorTransformation = lambda x: x,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        gen_optimizer: Optional[snt.Optimizer] = None,
        encoder_optimizer: Optional[snt.Optimizer] = None,
        disc_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        per: bool = False,
        importance_sampling_exponent: float = 0.2,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        checkpoint_folder: str = '',
        replay_client: Optional[Union[reverb.Client, reverb.TFClient]] = None,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          obj_func: objective function for policy gradient update. (var or cvar)
          critic_loss_type: c51 or qr.
          threshold: threshold for objective function 
          dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
          critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        if isinstance(replay_client, reverb.TFClient):
            replay_client = reverb.Client(replay_client._server_address)
        self._replay_client = replay_client
        self.per = per
        self.importance_sampling_exponent = importance_sampling_exponent
        self._th = threshold
        self._obj_func = obj_func
        if critic_loss_type == 'c51':
            self._critic_loss_func = losses.categorical
        elif (critic_loss_type == 'qr') or (critic_loss_type == 'iqn'):
            self._critic_loss_func = quantile_regression
        self._critic_type = critic_loss_type

        ## GAN params
        self.z_dim=z_dim
        self.decay_factor=decay_factor

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._gen_network=gen_network
        self._disc_network = disc_network
        self._encoder_loc_network = encoder_loc_network
        self._encoder_scale_network = encoder_scale_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network
        self._target_gen_network = gen_network
        self._target_disc_network = disc_network
        self._target_encoder_loc_network = encoder_loc_network
        self._target_encoder_scale_network = encoder_scale_network

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = tf2_utils.to_sonnet_module(
            observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Batch dataset and create iterator.
        self._base_iterator = dataset_iterator
        self._demo_iterator = demo_dataset_iterator
        self._run_demo = demo_dataset_iterator is not None
        self._demo_step = demo_step
        self._iterator = self._demo_iterator or self._base_iterator

        # Create optimizers if they aren't given.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._gen_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._disc_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._encoder_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

        # Expose the variables.
        policy_network_to_expose = snt.Sequential(
            [self._target_observation_network, self._target_policy_network])
        self._variables = {
            'critic': self._target_critic_network.variables,
            'policy': policy_network_to_expose.variables,
            'generator': self._target_gen_network.variables,
            'discriminator': self._target_disc_network.variables,
            'encoder_scale': self._target_encoder_scale_network.variables,
            'encoder_loc': self._target_encoder_loc_network.variables,
        }

        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                directory=checkpoint_folder,
                subdirectory='gan_learner',
                add_uid=False,
                objects_to_save={
                    'counter': self._counter,
                    'policy': self._policy_network,
                    'critic': self._critic_network,
                    'observation': self._observation_network,
                    'generator': self._gen_network,
                    'discriminator': self._disc_network,
                    'encoder_scale': self._encoder_scale_network,
                    'encoder_loc': self._encoder_loc_network,
                    'target_policy': self._target_policy_network,
                    'target_critic': self._target_critic_network,
                    'target_observation': self._target_observation_network,
                    'target_generator': self._target_gen_network,
                    'target_discriminator': self._target_disc_network,
                    'target_encoder_scale': self._target_encoder_scale_network,
                    'target_encoder_loc': self._target_encoder_loc_network,
                    'policy_optimizer': self._policy_optimizer,
                    'critic_optimizer': self._critic_optimizer,
                    'generator_optimizer':self._gen_optimizer,
                    'discriminator_optimizer': self._disc_optimizer,
                    'encoder_optimizer': self._encoder_optimizer,
                    'num_steps': self._num_steps,
                })
            gen_mean = snt.Sequential(
                [self._gen_network, acme_nets.StochasticMeanHead()])
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'policy': self._policy_network,
                    'generator': gen_mean,
                })

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        # Update target network
        online_variables = (
            *self._observation_network.variables,
            *self._critic_network.variables,
            *self._policy_network.variables,
            *self._gen_network.variables,
            *self._disc_network.variables,
            *self._encoder_loc_network.variables,
            *self._endcoder_scale_network.variables,
        )
        target_variables = (
            *self._target_observation_network.variables,
            *self._target_critic_network.variables,
            *self._target_policy_network.variables,
            *self._target_gen_network.variables,
            *self._target_disc_network.variables,
            *self._target_encoder_loc_network.variables,
            *self._target_endcoder_scale_network.variables,
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)
        transitions: types.Transition = sample.data  # Assuming ReverbSample.
        keys, probs = sample.info[:2]

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            # Maybe transform the observation before feeding into policy and critic.
            # Transforming the observations this way at the start of the learning
            # step effectively means that the policy and critic share observation
            # network weights.
            o_tm1 = self._observation_network(transitions.observation)
            o_t = self._target_observation_network(
                transitions.next_observation)
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tree.map_structure(tf.stop_gradient, o_t)

            batch_size = tf.shape(o_t)[0]
            # Generate z_target
            z = tf.random.normal(shape=(batch_size, self.z_dim))  ##z

            g_tm1 = self._generator_network(z, o_tm1, transitions.action).values  #### g(s)

            g_t= self._target_generator_network(z, o_t, self._target_policy_network(o_t)).values

            x_t= tf.reshape(transitions.reward, (-1,1))  \
                 + tf.reshape(discount * transitions.discount, (-1,1)) * g_t ## x= [batch_size, n_quantiles]  ##Bellman target

            ## target prior_encoder for r_int
            target_prior_encoder_loc = self._target_encoder_loc_network(g_tm1, o_tm1, transitions.action)
            target_prior_encoder_scale = self._target_encoder_scale_network(g_tm1, o_tm1, transitions.action)
            target_prior_encoder_dis = tfd.MultivariateNormalDiag(loc=target_prior_encoder_loc,
                                                                  scale_diag=target_prior_encoder_scale)

            ## target posterior_encoder  for r_int
            target_posterior_encoder_loc= self._target_encoder_loc_network (x_t, o_tm1, transitions.action)  ## x= [batch_size, self.z_dim]
            target_posterior_encoder_scale=self._target_encoder_scale_network (x_t, o_tm1, transitions.action)
            target_posterior_encoder_dis = tfd.MultivariateNormalDiag(loc=target_posterior_encoder_loc,
                                                                      scale_diag=target_posterior_encoder_scale)

            ## intrinsic reward
            r_intr= tfd.kl_divergence(target_posterior_encoder_dis, target_prior_encoder_dis)
            r_combined= tf.reshape(transitions.reward, (-1,1)) + self.decay_factor * tf.reshape(r_intr, (-1,1))

            # Critic learning.
            q_tm1 = self._critic_network(o_tm1, transitions.action)
            q_t = self._target_critic_network(
                o_t, self._target_policy_network(o_t))

            # Critic loss.
            critic_loss = self._critic_loss_func(q_tm1, r_combined,
                                                 discount * transitions.discount, q_t)

            ##  encoder sampling
            encoder_loc = self._encoder_loc_network(x_t, o_tm1, transitions.action)  ## [batch_size, self.z_dim]
            encoder_scale = self._encoder_scale_network (x_t, o_tm1, transitions.action)
            encoder_dis = tfd.MultivariateNormalDiag(loc=encoder_loc, scale_diag=encoder_scale)

            encoder_samples = encoder_dis.sample()  ## z_tilde should be [batch_size, self.z_dim]
            encoder_samples = tf.squeeze(encoder_samples, axis=0)

            z = tf.random.normal(shape=(batch_size, self.z_dim))  ##z
            generator_samples = self._generator_network(z, o_tm1, transitions.action).values  #### x_tilde

            # discriminator loss
            disc_output_1= self._disc_network(generator_samples, z, o_tm1, transitions.action )
            disc_output_2 = self._disc_network(x_t, encoder_samples, o_tm1, transitions.action)
            disc_loss= -tf.reduce_mean(tf.log(disc_output_1)+ tf.log(1-disc_output_2 ), (0, -1))

            ### encoder loss
            encoder_loss = -tf.reduce_mean(tf.log(1-disc_output_1) + tf.log(disc_output_2), (0, -1))

            # gen loss (mean of squared differences)
            squared_diff = tf.pow(x_t - self._generator_network(encoder_samples, o_tm1, transitions.action), 2)
            gen_loss = tf.reduce_mean(squared_diff, (0, -1))

            ## actor loss and learning
            # Policy loss
            if self._run_demo:
                # Policy loss MSE supervised learning
                dpg_a_t = self._policy_network(o_t)
                policy_loss = tf.reduce_mean(tf.square(dpg_a_t - transitions.action),
                                             axis=[0])
            else:
                # Actor learning.
                dpg_a_t = self._policy_network(o_t)
                if self._critic_type == 'iqn':
                    dpg_z_t = self._critic_network(o_t, dpg_a_t, policy=True)
                else:
                    dpg_z_t = self._critic_network(o_t, dpg_a_t)

                if self._obj_func == 'meanstd':
                    dpg_q_t = dpg_z_t.meanstd()
                elif self._obj_func == 'var':
                    dpg_q_t = dpg_z_t.var(self._th)
                elif self._obj_func == 'cvar':
                    if self._critic_type == 'qr':
                        dpg_q_t = dpg_z_t.cvar(self._th)
                    elif self._critic_type == 'iqn':
                        dpg_q_t = dpg_z_t.mean()
                elif self._obj_func == 'gain_loss_tradeoff':
                    dpg_q_t = dpg_z_t.gain_loss_tradeoff()

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._clipping else None
                policy_loss = losses.dpg(
                    dpg_q_t,
                    dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping)
                policy_loss = tf.reduce_mean(policy_loss, axis=[0])


            if self.per:
                # PER: importance sampling weights.
                priorities = tf.abs(critic_loss)
                importance_weights = 1.0 / probs
                importance_weights **= self.importance_sampling_exponent
                importance_weights /= tf.reduce_max(importance_weights)
                critic_loss *= tf.cast(importance_weights,
                                       critic_loss.dtype)


        # Get trainable variables.
        policy_variables = self._policy_network.trainable_variables

        gen_variables = (
            # In this agent, the critic loss trains the observation network.
            self._observation_network.trainable_variables +
            self._generator_network.trainable_variables
            )

        critic_variables = self._critic_network.trainable_variables

        disc_variables = self._disc_network.trainable_variables
        encoder_variables = (self._encoder_loc_network +
                             self._encoder_scale_network)

        # Compute gradients.
        policy_gradients = tape.gradient(policy_loss, policy_variables)
        critic_gradients = tape.gradient(critic_loss, critic_variables)
        disc_gradients = tape.gradient(disc_loss, disc_variables)
        gen_gradients = tape.gradient(gen_loss, gen_variables)
        encoder_gradients = tape.gradient(encoder_loss, encoder_variables)

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
            critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]
            disc_gradients = tf.clip_by_global_norm(disc_gradients, 40.)[0]
            gen_gradients = tf.clip_by_global_norm(gen_gradients, 40.)[0]
            encoder_gradients = tf.clip_by_global_norm(encoder_gradients, 40.)[0]

        # Apply gradients.
        self._critic_optimizer.apply(critic_gradients, critic_variables)
        self._disc_optimizer.apply(disc_gradients, disc_variables)
        self._encoder_optimizer.apply(encoder_gradients, encoder_variables)
        self._gen_optimizer.apply(gen_gradients, gen_variables)
        self._policy_optimizer.apply(policy_gradients, policy_variables)


        # Losses to track.
        return {
            'critic_loss': critic_loss,
            'discriminator_loss': disc_loss,
            'generator_loss': gen_loss,
            'encoder_loss': encoder_loss,
            'policy_loss': policy_loss,
            'keys': keys,
            'priorities': priorities if self.per else None,
        }

    def step(self):
        # Run the learning step.
        if self._num_steps > self._demo_step:
            self._iterator = self._base_iterator
            self._run_demo = False

        fetches = self._step()

        if self.per:
            # Update priorities in replay.
            keys = fetches.pop('keys')
            priorities = fetches.pop('priorities')
            if self._replay_client:
                self._replay_client.mutate_priorities(
                    table=reverb_adders.DEFAULT_PRIORITY_TABLE,
                    updates=dict(zip(keys.numpy(), priorities.numpy())))
        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
