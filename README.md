# RL-Hedging

Application for Option Portfolio Hedging

This application consists of two parts: an environment simulator and a D4PG-based RL algorithm.

1. Environment Simulator:
    - It simulates real-world option market maker's trading environment.
    - It provides a realistic trading environment for training and testing option hedging strategies.
    - Environment uses OpenAI Gym interface.

2. D4PG-based RL Algorithm:
    - The RL algorithm is based on the D4PG (Distributed Distributional Deterministic Policy Gradient) algorithm.
    - RL implementation follow Deepmind ACME framework.

## Table of Contents

- [Introduction](#introduction)
- [Entrypoints](#entrypoints)
- [Modules](#modules)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

[Provide a more detailed introduction to the package, its scope, and its primary features.]

## Entrypoints

### `run_benchmark.py`

- Description: Used to run BSM hedging benchmark algorithms.
- Usage: [Provide example command or syntax]

### `run_d3pg.py`

- Description: Implements Distributional Deep Deterministic Policy Gradients (D3PG), a distributional RL algorithm.
- Usage: [Include usage instructions]

### `run_d4pg.py`

- Description: Focuses on Distributed Distributional Deep Deterministic Policy Gradients (D4PG) with distributed actors.
- Usage: [Offer a usage example]

## Modules

### Environment Module (`env`)

- Description: Implements environments using the OpenAI Gym interface.
- Key Features: [Elaborate on key features]

### Agent Module (`agent`)

- Description: Contains the RL algorithms.
- Key Features: [Detail specific algorithms or implementations]

### Domain Module (`domain`)

#### SDE Module

- Description: Simulates stock prices and implied volatilities.

#### Asset Module

- Description: Handles various assets such as stocks, options, and portfolios.

#### Pricer Module

- Description: Responsible for asset pricing.

### Domain Randomization Module (`dr`)

- Description: Supports domain randomization for robust training.

### Configuration Module (`config`)

- Description: Used for constructing environments with YAML configurations.

### Analysis Module

- Description: Generates evaluation metrics for the models.

## Installation

[Provide detailed instructions for installing the package, including dependencies.]

## Usage Examples

[Include examples of how to use the main modules and entrypoints. Code snippets or command-line instructions can be very helpful.]

## Contributing

[Guidelines for those who wish to contribute to the project.]

## License

[Specify the licensing information for your package.]
