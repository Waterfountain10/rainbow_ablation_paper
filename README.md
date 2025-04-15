# Code your parts accordingly

- denis : 1 and 2
- william : 3,4 and 5
- max : 6, 7, 8

agents done implenting up until now:
- dqn
- ddqn
- prioritized
- duel



## Merging ideas
ddqn -> if on : Doing next on both target and live network (compute loss)
         else : updates using live network for loss

multi-step -> if on : uses NstepBuffer instead of replay
               else : use replay

Prioritized -> if on : overwrites memory using different buffer, params, weighted loss (update model), PER specific beta compute loss (reduction), (modify beta in training)
                else : 

DuelNet -> if on : Different network (value)

Noisy -> if on : Different network (different type of layer), (no epsilon), update target and live (update model)

categorical -> if on : network = categoricaldqn network (needs another param)
                       loss computation uses projection and cross entropy
                else : network = normal dqn network
                       stays with MSE loss
