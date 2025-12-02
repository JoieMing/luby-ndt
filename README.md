# A Differentiable Digital Twin of Distributed Link Scheduling for Contention-Aware Networking
## Abstract
Many routing and flow optimization problems in wired networks can be solved efficiently using minimum cost flow formulations. 
However, this approach does not extend to wireless multi-hop networks, where the assumptions of fixed link capacity and linear cost structure collapse due to contention for shared spectrum resources. 
The key challenge is that the long-term capacity of a wireless link becomes a non-linear function of its network context, including network topology, link quality, and the traffic assigned to neighboring links. 
In this work, we pursue a new direction of modeling wireless network under randomized medium access control by developing an analytical network digital twin (NDT) that predicts link duty cycles from network context. 
We generalize randomized contention as finding a Maximal Independent Set (MIS) on the conflict graph using weighted Luby's algorithm, derive an analytical model of link duty cycles, and introduce an iterative procedure that resolves the circular dependency among duty cycle, link capacity, and contention probability. 
Our numerical experiments show that the proposed NDT accurately predicts link duty cycles and congestion patterns with up to a 5000× speedup over packet-level simulation, and enables us to optimize link scheduling using gradient descent for reduced congestion and radio footprint.
## Instructions for replicate numerical experiments
