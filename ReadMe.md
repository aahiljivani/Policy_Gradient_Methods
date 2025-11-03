# **REINFORCE & REINFORCE with Baseline – Policy Gradient Methods Demo**
# Introduction

This project demonstrates the implementation and training results for the REINFORCE and REINFORCE with Baseline algorithms. Both are classic policy gradient methods in reinforcement learning, aiming to directly optimize the expected value of the policy by adjusting its parameters via gradient ascent.

Below, I've included pseudocode images for each method to illustrate their key conceptual steps. These algorithms work by breaking down the gradient of the value function 
Vπ(s) into the expected state-action value, and then, through using the likelihood-ratio trick, arrive at an expression where the gradient is sampled via returns and the policy's log-probabilities.


REINFORCE is a Monte Carlo policy gradient algorithm that updates the policy using episode returns. While it is simple and works in many situations, it suffers from high variance in the gradient estimates, which can make learning unstable or slow. For the policy network, I use three fully connected layers, with two ReLU activations, and a final softmax output to parameterize the policy over actions.

REINFORCE with Baseline introduces an unbiased estimate of the value function as a baseline (implemented as a neural network with two ReLU-activated hidden layers and a final scalar output per batch). This baseline helps reduce the variance by estimating how "good" taking a certain action in a given state is compared to the average for that state—this difference is called the advantage function.
