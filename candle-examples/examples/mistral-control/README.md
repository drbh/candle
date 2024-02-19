# mistral-control

This example demonstrates how to guide Mistral's output via a control vector.

control vectors are a set of values that are used during the generation to direct the output towards conecpts/topics that have been captured in the control vector.

technically the control vector is simply a matrix with the shape (number_of_layers, hidden_size) where each row is a control vector for a layer and it is simply added to the hidden states of the layer.

## Usage

create data and capture hidden states:

```bash
cargo run \
--example mistral-control \
--features cuda \
--release \
-- \
--model-id mistralai/Mistral-7B-Instruct-v0.1 \
--base-prompt "<s> [INST] Act as if you're extremely" \
--positive-adjectives "happy [/INST]" \
--negative-adjectives "sad [/INST]" \
--dump-last-hidden-states
```

reduce the hidden states to a control vector via PCA

```bash
python candle-examples/examples/mistral-control/generate.py
```

apply a control vector to a prompt:

```bash
cargo run \
--example mistral-control \
--features cuda \
--release \
-- \
--model-id mistralai/Mistral-7B-Instruct-v0.1 \
--prompt "How do you feel today? " \
--sample-len 64 \
--seed 0 \
--repeat-penalty 1.3 \
--control-vector-path-sf directions.safetensors  \
--control-strength 1.5 \
--apply-control-vector

# How do you feel today?  Are you feeling great, awesome and excited! Or are you smiling with a big smile on your face.

# I am so happy to be able to share this beautiful energy of the day with all my friends and family around the world. I love it when we get together in our amazing community and celebrate each other’
```

```bash
cargo run \
--example mistral-control \
--features cuda \
--release \
-- \
--model-id mistralai/Mistral-7B-Instruct-v0.1 \
--prompt "How do you feel today? " \
--sample-len 64 \
--seed 0 \
--repeat-penalty 1.3 \
--control-vector-path-sf directions.safetensors  \
--control-strength -1.5 \
--apply-control-vector

# How do you feel today?  Are your feelings of despair, hopelessness and depression ruining everything in life for you.

# If so then this article is not going to be a waste of time reading it.

# I have been depressed since I was about the age of 14 (20 years old now). It has ruined
```

## Resources

original paper <https://www.ai-transparency.org/>
inspiring blog post: <https://vgel.me/posts/representation-engineering/>
householder <https://www.cs.cornell.edu/%7Ebindel/class/cs6210-f12/notes/lec16.pdf>
householder and qr: <https://stackoverflow.com/a/53493770>
schur factorization: <https://en.wikipedia.org/wiki/Schur_decomposition>
qr to eigenvalues and eigenvectors: <https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition>
notebook of similar: <https://github.com/vgel/repeng/blob/main/notebooks/experiments.ipynb>
