"""
https://arxiv.org/pdf/1905.11946
Contribution: Proposed a new scaling method that uniformly scales all dimensions of depth, width, and resolution
using a compound coefficient, resulting in more efficient models.

Observation 1 – Scaling up any dimension of network width, depth,
or resolution improves accuracy, but the accuracy gain diminishes for bigger models
Observation 2 – In order to pursue better accuracy and efficiency, it is critical
to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

-> compound scaling method,
- which use a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way:

"""