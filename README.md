# EasyDreamer: A Simplified Version of the Dreamer Algorithm with Pytorch

## Introduction

In this repository, we've implemented a simplified version of the Dreamer algorithm, which is explained in detail in the paper [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). The main goal of Dreamer is to train a model that helps agents perform well in environments with high sample efficiency. We have implemented our version of Dreamer using PyTorch, which simplifies the process and makes the model more accessible to researchers and practitioners who are already familiar with the PyTorch framework. With this implementation, they can gain a deeper understanding of how the algorithm works and test their own ideas more efficiently, contributing to the advancement of research in this field.

We have also included a re-implementation of Plan2Explore, a model-based exploration method introduced in the paper [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960). Plan2Explore is designed to improve generalization about the model without any task-relevant information by using an unsupervised learning approach. Our PyTorch implementation of Plan2Explore is available in this repository.

### Differences from other implementations

Our implementation of Dreamer differs from others in several ways. Firstly, we separate the recurrent model from the other models to gain a better understanding of how deterministic processing works. Secondly, we align the naming conventions used in our implementation with those in the paper. Furthermore, modules are trained following the same pseudo code as outlined in the original Dreamer paper. Thirdly, we remove overshooting, which was crucial in Dreamer-v1 and model-based approaches but is no longer mentioned in Dreamer-v2 and v3, and is even omitted from official implementations. Lastly, we use a single-step lambda value calculation, which enhances readability at the expense of performance.


<hr/>

## Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

<hr/>

## run

To run the training process, use the following command:

### Dreamer
```
python main.py --config dmc-walker-walk
```
### Plan2Explore
```
python main.py --config p2e-dmc-walker-walk
```
<hr/>

## Architecture

### Dreamer
```
┌── dreamer
│   ├── algorithms
│   │   └── dreamer.py : Dreamer algorithm. Including the loss function and training loop
│   │   └── plan2explore.py : plan2explore algorithm. Including the loss function and training loop
│   ├── configs
│   │   └ : Contains hyperparameters for the training process and sets up the training environment
│   ├── envs
│   │   ├── envs.py : Defines the environments used in the Dreamer algorithm
│   │   └── wrappers.py : Modifies observations of the environments to make them more suitable for training
│   ├── modules
│   │   ├── actor.py : A linear network to generate action
│   │         └ input : deterministic and stochastic(state)
│   │         └ output : action
│   │   ├── critic.py : A linear network to generate value
│   │         └ input : deterministic and stochastic
│   │         └ output : value
│   │   ├── decoder.py : A convTranspose network to generate reconstructed image
│   │         └ input : deterministic and stochastic
│   │         └ output : reconstructed image
│   │   ├── encoder.py : A convolution network to generate embedded observation
│   │         └ input : image
│   │         └ output : embedded observation
│   │   ├── model.py : Contains the implementation of models
│   │       └ RSSM : Stands for "Recurrent State-Space Model"
│   │         └ RecurrentModel : A recurrent neural network to generate deterministic.
│   │           └ input : stochastic and deterministic and action
│   │           └ output : deterministic
│   │         └ TransitionModel : A linear network to generate stochastic. we call it as prior
│   │           └ input : deterministic
│   │           └ output : stochastic(prior)
│   │         └ RepresentationModel : A linear network to generate stochastic. we call it as posterior.
│   │           └ input : embedded observation and deterministic
│   │           └ output : stochastic(posterior)
│   │       └ RewardModel : A linear network to generate reward
│   │         └ input : deterministic and stochastic 
│   │         └ output : reward
│   │       └ ContinueModel : A linear network to generate continue flag(not done)
│   │         └ input : deterministic and stochastic
│   │         └ output : continue flag
│   │   └── one_step_model.py : A linear network to predict embedded observation # for plan2explore
│   │         └ input : deterministic and stochastic and action
│   │         └ output : embedded observation
│   └── utils
│       ├── buffer.py : Contains the replay buffer used to store and sample transitions during training
│       └── utils.py : Contains other utility functions
└── main.py : Reads the configuration file, sets up the environment, and starts the training process
```

<hr/>

## Todo

* discrete action space environment performance check
* code-coverage test
* dreamer-v2
* dreamer-v3

<hr/>

## Performance

### Dreamer

| Task                    | 20-EMA  |
|-------------------------|--------|
| ball-in-cup-catch        | 936.9  |
| walker-stand             | 972.8  |
| quadruped-walk           | 584.7  |
| cheetah-run              | 694.0  |
| cartpole-balance         | 831.2  |
| cartpole-swingup-sparse  | 219.3  |
| finger-turn_easy         | 805.1  |
| cartpole-balance-sparse  | 541.6  |
| hopper-hop               | 250.7  |
| walker-run               | 284.6  |
| reacher-hard             | 162.7  |
| reacher-easy             | 911.4  |
| acrobot-swingup          | 91.8   |
| finger-spin              | 543.5  |
| cartpole-swingup         | 607.8  |
| walker-walk              | 871.3  |

All reported results were obtained by running the experiments 3 times with different random seeds. Evaluation was performed after each interaction with the environment, and the reported performance metric is the 20-EMA (exponential moving average) of the cumulative reward in a single episode.

<hr/>

## References

* [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
* [https://github.com/google-research/dreamer](https://github.com/google-research/dreamer)
* [https://github.com/danijar/dreamer](https://github.com/danijar/dreamer)
* [https://github.com/juliusfrost/dreamer-pytorch](https://github.com/juliusfrost/dreamer-pytorch)
* [https://github.com/yusukeurakami/dreamer-pytorch](https://github.com/yusukeurakami/dreamer-pytorch)
* [Planning to Explore via Self-Supervised World Models](https://arxiv.org/abs/2005.05960)
* [https://github.com/danijar/dreamerv2](https://github.com/danijar/dreamerv2)
