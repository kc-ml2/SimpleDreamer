# SimpleDreamer: A Simplified Version of the Dreamer Algorithm

## Introduction

In this repository, we've implemented a simplified version of the Dreamer algorithm, which is explained in detail in the paper [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603). Dreamer's main goal is to train a model that helps agents perform well in environments with high sample efficiency. Our implementation aims to provide researchers and practitioners with a simplified version of Dreamer, allowing them to gain a deeper understanding of how the algorithm works and test their own ideas more efficiently.

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

```
python main.py --config dmc-walker-walk
```

<hr/>

## Performance

* Will be updated

<hr/>

## Architecture
```
┌── dreamer
│   ├── algorithms
│   │   └── dreamer.py : Dreamer algorithm. Including the loss function and training loop
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
│   │   └── model.py : Contains the implementation of models
│   │       └ RSSM : Stands for "Recurrent State-Space Model"
│   │         └ RecurrentModel : A recurrent neural network to generate deterministic.
│   │           └ input : stochastic and deterministic and action
│   │           └ output : deterministic
│   │         └ TransitionModel : A linear network to generate stochastic. we call it as prior
│   │           └ input : deterministic
│   │           └ output : stochastic(prior)
│   │         └ RepresentationModel : A linear network to generate stochastic. we call it as posterior.
│   │           └ input : embedded observation and deterministic and action 
│   │           └ output : stochastic(posterior)
│   │       └ RewardModel : A linear network to generate reward
│   │         └ input : deterministic and stochastic 
│   │         └ output : reward
│   │       └ ContinueModel : A linear network to generate continue flag(not done)
│   │         └ input : deterministic and stochastic
│   │         └ output : continue flag
│   └── utils
│       ├── buffer.py : Contains the replay buffer used to store and sample transitions during training
│       └── utils.py : Contains other utility functions
└── main.py : Reads the configuration file, sets up the environment, and starts the training process
```
<hr/>

## Todo

* discrete action space environment performance check
* code-coverage test
* plan2explore
* dreamer-v2
* dreamer-v3

<hr/>

## References

* [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
* [https://github.com/google-research/dreamer](https://github.com/google-research/dreamer)
* [https://github.com/danijar/dreamer](https://github.com/danijar/dreamer)
* [https://github.com/juliusfrost/dreamer-pytorch](https://github.com/juliusfrost/dreamer-pytorch)
* [https://github.com/yusukeurakami/dreamer-pytorch](https://github.com/yusukeurakami/dreamer-pytorch)



