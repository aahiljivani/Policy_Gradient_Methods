# **REINFORCE & REINFORCE with Baseline – Policy Gradient Methods Demo**
# Introduction

This project demonstrates the implementation and training results for the REINFORCE and REINFORCE with Baseline algorithms. Both are classic policy gradient methods in reinforcement learning, aiming to directly optimize the expected value of the policy by adjusting its parameters via gradient ascent.

Below, I've included pseudocode images for each method to illustrate their key conceptual steps. These algorithms work by breaking down the gradient of the value function 
Vπ(s) into the expected state-action value, and then, through using the likelihood-ratio trick, arrive at an expression where the gradient is sampled via returns and the policy's log-probabilities as shown below:

<img width="1481" height="407" alt="Screenshot 2025-11-02 at 11 19 52 PM" src="https://github.com/user-attachments/assets/c00d75ae-bdd9-4d31-922f-d9b05f8a5b16" />


REINFORCE is a Monte Carlo policy gradient algorithm that updates the policy using episode returns. While it is simple and works in many situations, it suffers from high variance in the gradient estimates, which can make learning unstable or slow. For the policy network, I use three fully connected layers, with two ReLU activations, and a final softmax output to parameterize the policy over actions.

<img width="1562" height="569" alt="Screenshot 2025-11-02 at 11 20 13 PM" src="https://github.com/user-attachments/assets/57f06f56-6100-4349-88d0-4b4f04d22ab0" />


REINFORCE with Baseline introduces an unbiased estimate of the value function as a baseline (implemented as a neural network with two ReLU-activated hidden layers and a final scalar output per batch). This baseline helps reduce the variance by estimating how "good" taking a certain action in a given state is compared to the average for that state—this difference is called the advantage function.

<img width="1566" height="722" alt="Screenshot 2025-11-02 at 11 20 39 PM" src="https://github.com/user-attachments/assets/33108df9-9be3-48bc-ac80-b2b5f362f39b" />

Below are average reward plots over training episodes for both approaches on CartPole-v1:

REINFORCE (without baseline): Shows that although the algorithm can converge for 10 episodes on the highest score attainable, this is followed by major dips in training performance, higher variance in the returns, with more fluctuations and slower improvement.

<img width="601" height="456" alt="Screenshot 2025-11-02 at 11 24 29 PM" src="https://github.com/user-attachments/assets/0a1673ce-c299-416a-b007-e926adaa21b3" />

REINFORCE with Baseline: Shows smoother convergence towards optimal performance, thanks to the variance reduction brought by the value function network. we see consistent max rewards attained over episodes after 400 with some slight dips which are then reconciled through max score trends again. Although the attainment of max scores is slower, we can visually see lower variance in the training graph with more noise in the midrange during the middle of training.

<img width="594" height="448" alt="Screenshot 2025-11-02 at 11 36 15 PM" src="https://github.com/user-attachments/assets/281ce0cd-462c-4345-bf66-fcb865d562e3" />

