<div align="center">
    <h1><code>`mnist-from-scratch`</code></h1>
    <i>A handwritten digit classsifier neural network written and trained from scratch in pure zig</i>
    <br/>
    Uses <code>brainz</code> library for model training.
    <hr>
</div>

![UI Screenshot](assets/ui.png)


## Getting it up running

This requires Zig **0.13.0** to work.

1. Clone the git repository
2. Download and extract the MNIST dataset into the repository root.
2. Run `zig build`
3. Deployable WASM + HTML file are at `zig-out/web/public`

## Training the model locally

You can train the model locally by running `zig build train`. By default the training will use the model checkpoints in `src/model.bin`. You can start from scratch by deleting the checkpoints and running the command again.


## License and acknowledgements

This project is licensed under **MIT licence**.

Uses the MNIST dataset (https://yann.lecun.com/exdb/mnist/)

