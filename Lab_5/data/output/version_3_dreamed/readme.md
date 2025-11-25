# Parameters for version 3
iterations = 30
lr = 0.008
num_octaves = 3
octave_scale = 1.2
layer_names_weights = {'inception4a': 0.3, 'inception4e': 0.7}

Additionally, jitter was reduced to 16 here.

Observation: This version, there was visible fo original image shapes patterns, and the hallucinated eyes with hypnotic circluar pattern was less and was present at the corners of the shapes of original image. Additionally, there was other triangular shaped pattern and together both, the dreamed image was looking better than the previous versions.