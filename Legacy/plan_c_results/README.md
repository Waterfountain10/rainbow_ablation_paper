### We are hoping to store results of **stock-ex** or some other stock environement, here, as numpy array files (.npy).

```bash
<model_name>_results.npy
```

They would be used for further plotting.

### ... but for now:

we are temporarily storing **CartPole** results with standard parameters:

```python
#Parameters for DQN
    MEMORY_SIZE = 20000
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    EPSILON_DECAY_STEPS = 1500
    LEARNING_RATE = 5e-4
    NUM_EPISODES = 2000  # Small number for testing (increased it to compare with PER - will)
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make("CartPole-v1")
```
