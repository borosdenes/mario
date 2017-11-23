from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import time

from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# ---------------------------------------------------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------------------------------------------------

# python examples/openai_gym.py Pong-ram-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 50000 -m 2000
# python examples/openai_gym.py CartPole-v0 -a examples/configs/vpg.json -n examples/configs/mlp2_network.json -e 2000 -m 200
# python mario.py Pong-v0 -a /home/deeplearning/tensorforce/examples/configs/vpg.json -n cnn_lstm_network.json -e 50000 -m 2000
# python mario.py Pong-v0 -a vpg.json -n cnn_lstm_network.json -e 50000 -m 2000 --monitor ./monitor --monitor-video 100 --debug

# ---------------------------------------------------------------------------------------------------------------------------------------

def main():

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# ARGPARSE
	# ---------------------------------------------------------------------------------------------------------------------------------------
	parser = argparse.ArgumentParser()

	parser.add_argument('gym_id', help="Id of the Gym environment")
	parser.add_argument('-a', '--agent-config', help="Agent configuration file")
	parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
	parser.add_argument('-e', '--episodes', type=int, default=None, help="Number of episodes")
	parser.add_argument('-t', '--timesteps', type=int, default=None, help="Number of timesteps")
	parser.add_argument('-m', '--max-episode-timesteps', type=int, default=None, help="Maximum number of timesteps per episode")
	parser.add_argument('-d', '--deterministic', action='store_true', default=False, help="Choose actions deterministically")
	parser.add_argument('-l', '--load', help="Load agent from this dir")
	parser.add_argument('--monitor', help="Save results to this directory")
	parser.add_argument('--monitor-safe', action='store_true', default=False, help="Do not overwrite previous results")
	parser.add_argument('--monitor-video', type=int, default=0, help="Save video every x steps (0 = disabled)")
	parser.add_argument('-D', '--debug', action='store_true', default=False, help="Show debug outputs")

	args = parser.parse_args()

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# LOGGER
	# ---------------------------------------------------------------------------------------------------------------------------------------
	
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)
	
	# ---------------------------------------------------------------------------------------------------------------------------------------
	# ENVIRONMENT
	# ---------------------------------------------------------------------------------------------------------------------------------------

	environment = OpenAIGym(
		gym_id=args.gym_id,
		monitor=args.monitor,
		monitor_safe=args.monitor_safe,
		monitor_video=args.monitor_video
	)

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# LOAD AGENT CONFIGURATION
	# ---------------------------------------------------------------------------------------------------------------------------------------

	# config_vpg_agent_visual = Configuration.from_json('/Users/borosdenes/tensorforce/examples/configs/vpg_agent_visual.json')
	# config_vpg_agent = Configuration.from_json('/Users/borosdenes/tensorforce/examples/configs/vpg_agent.json')

	# ---------------------------------------------------------------------------------------------------------------------------------------

	if args.agent_config is not None:
		with open(args.agent_config, 'r') as fp:
			agent_config = json.load(fp=fp)
	else:
		raise TensorForceError("No agent configuration provided.")

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# CREATE PREPROCESSING
	# ---------------------------------------------------------------------------------------------------------------------------------------

	# preprocessing_config = [
	# 	{
	# 		"type": "image_resize",
	# 		"kwargs": {
	# 			"width": 84,
	# 			"height": 84
	# 		}
	# 	},	{
	# 		"type": "grayscale"
	# 	},	{
	# 		"type": "center"
	# 	}, 	{
	# 		"type": "sequence",
	# 		"kwargs": {
	# 			"length": 4
	# 		}
	# 	}
	# ]

	# stack = Preprocessing.from_spec(preprocessing_config)
	# config.state_shape = stack.shape(config.state_shape)

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# NETWORK SPECIFICAITON
	# ---------------------------------------------------------------------------------------------------------------------------------------

	if args.network_spec is not None:
		with open(args.network_spec, 'r') as fp:
			network_spec = json.load(fp=fp)
	else:
		network_spec = None
		logger.info("No network configuration provided.")

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# AGENT SPECIFICAITON
	# ---------------------------------------------------------------------------------------------------------------------------------------

	agent = Agent.from_spec(
		spec=agent_config,
		kwargs=dict(
			states_spec=environment.states,
			actions_spec=environment.actions,
			network_spec=network_spec
		)
	)

	# ---------------------------------------------------------------------------------------------------------------------------------------

	# agent = VPGAgent(
	# 	states_spec=environment.states,
	# 	actions_spec=environment.actions,
	# 	network_spec=network_spec,
	# 	batch_size=64
	# )

	# ---------------------------------------------------------------------------------------------------------------------------------------

	# vpg_agent = Agent.from_spec(
	# 	spec='vpg_agent',
	# 	kwargs=dict(
	# 		states_spec=environment.states,
	# 		actions_spec=environment.actions,
	# 		network_spec=network_spec,
	# 		# preprocessing=preprocessing_config,
	# 		config=config_vpg_agent_visual
	# 	)
	# )

	# ---------------------------------------------------------------------------------------------------------------------------------------

	if args.load:
		load_dir = os.path.dirname(args.load)
		if not os.path.isdir(load_dir):
			raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
		agent.restore_model(args.load)

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# RUNNER SPECIFICAITON
	# ---------------------------------------------------------------------------------------------------------------------------------------

	if args.debug:
		logger.info("-" * 16)
		logger.info("Configuration:")
		logger.info(agent_config)

	# ---------------------------------------------------------------------------------------------------------------------------------------

	runner = Runner(
		agent=agent,
		environment=environment,
		repeat_actions=1
	)

	# ---------------------------------------------------------------------------------------------------------------------------------------

	if args.debug:  # TODO: Timestep-based reporting
		report_episodes = 1
	else:
		report_episodes = 100

		logger.info("Starting {agent} for Environment '{env}'".format(agent=agent, env=environment))

	def episode_finished(r):
		if r.episode % report_episodes == 0:
			steps_per_second = r.timestep / (time.time() - r.start_time)
			logger.info("Finished episode {} after {} timesteps. Steps Per Second {}".format(
				r.agent.episode, r.episode_timestep, steps_per_second
			))
			logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
			logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / min(500, len(r.episode_rewards))))
			logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / min(100, len(r.episode_rewards))))

		return True

	# ---------------------------------------------------------------------------------------------------------------------------------------
	# RUN
	# ---------------------------------------------------------------------------------------------------------------------------------------
	runner.run(
		timesteps=args.timesteps,
		episodes=args.episodes,
		max_episode_timesteps=args.max_episode_timesteps,
		deterministic=args.deterministic,
		episode_finished=episode_finished
	)

	logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.agent.episode))

if __name__ == '__main__':
	main()
