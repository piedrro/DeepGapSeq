import numpy as np
from pomegranate import *

# Define the HMM model
model = HiddenMarkovModel()

# Define the states
state0 = State(NormalDistribution(0, 1), name="state0")
state1 = State(NormalDistribution(5, 1), name="state1")

# Add the states to the model
model.add_states(state0, state1)

# Define the transitions
model.add_transition(model.start, state0, 0.5)
model.add_transition(model.start, state1, 0.5)
model.add_transition(state0, state0, 0.7)
model.add_transition(state0, state1, 0.3)
model.add_transition(state1, state0, 0.3)
model.add_transition(state1, state1, 0.7)

# Finalize the model
model.bake()

# Generate some time series data
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# Fit the model to the data
model.fit([data])

# Print the learned parameters
# print("Initial probabilities:", model.start_probability)
print("Transition probabilities:", model.dense_transition_matrix())
print("Means:", [state.distribution.parameters[0] for state in model.states])
print("Standard deviations:", [state.distribution.parameters[1] for state in model.states])
