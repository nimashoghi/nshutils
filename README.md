# nshutils

`nshutils` is a collection of utility functions and classes that I've found useful in my day-to-day work as an ML researcher. This library includes utilities for typechecking, logging, and saving/loading activations from neural networks.

## Installation

To install `nshutils`, simply run:

```bash
pip install nshutils
```

## Features

### Typechecking

`nshutils` provides a simple way to typecheck your code using the [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) library. Simply call `typecheck_this_module()` at the top of your module (i.e., in the root `__init__.py` file) to enable typechecking for the entire module:

```python
from nshutils.typecheck import typecheck_this_module

typecheck_this_module()
```

You can also use the `tassert` function to assert that a value is of a certain type:

```python
import nshutils.typecheck as tc

def my_function(x: tc.Float[torch.Tensor, "bsz seq len"]) -> tc.Float[torch.Tensor, "bsz seq len"]:
    tc.tassert(tc.Float[torch.Tensor, "bsz seq len"], x)
    ...
```

### Logging

`nshutils` provides a simple way to configure logging for your project. Simply call one of the logging setup functions:

```python
from nshutils.logging import init_python_logging

init_python_logging()
```

This will configure logging to use pretty formatting for PyTorch tensors and numpy arrays (inspired by and/or utilizing [`lovely-numpy`](https://github.com/xl0/lovely-numpy) and [`lovely-tensors`](https://github.com/xl0/lovely-tensors)), and will also enable rich logging if the `rich` library is installed.

### Activation Saving/Loading

`nshutils` provides a simple way to save and load activations from neural networks. To save activations, use the `ActSave` object:

```python
from nshutils import ActSave

def my_model_forward(x):
    ...
    # Save activations to "{save_dir}/encoder.activations/{idx}.npy"
    ActSave({"encoder.activations": x})

    # Equivalent to the above
    with ActSave.context("encoder"):
        ActSave(activations=x)
    ...

ActSave.enable(save_dir="path/to/activations")
x = torch.randn(...)
my_model_forward(x)
# Activations are saved to disk under the "path/to/activations" directory
```

This will save the `x` tensor to disk under the `encoder` prefix.

To load activations, use the `ActLoad` class:

```python
from nshutils import ActLoad

act_load = ActLoad.from_latest_version("path/to/activations")
encoder_acts = act_load["encoder"]

for act in encoder_acts:
    print(act.shape)
```

This will load all of the activations saved under the `encoder` prefix.

### Other Utilities

`nshutils` also provides a few other utility functions/classes:

- `snoop`: A simple way to debug your code using the `pysnooper` library, based on the [`torchsnooper`](https://github.com/zasdfgbnm/TorchSnooper) library.
- `apply_to_collection`: Recursively apply a function to all elements of a collection that match a certain type, taken from the [`pytorch-lightning`](https://github.com/Lightning-AI/pytorch-lightning) library.

## Contributing

Contributions to `nshutils` are welcome! Please open an issue or submit a pull request on the [GitHub repository](https://github.com/nimashoghi/nshutils).

## License

`nshutils` is released under the MIT License. See the `LICENSE` file for more details.
