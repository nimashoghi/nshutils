# nshutils

`nshutils` is a collection of utility functions and classes that I've found useful in my day-to-day work as an ML researcher. This library includes utilities for typechecking, logging, and saving/loading activations from neural networks.

## Installation

To install `nshutils`, simply run:

```bash
pip install nshutils
```

## Configuration

`nshutils` features a unified configuration system that allows you to control various aspects of the library through environment variables, programmatic settings, or JSON configuration.

### Environment Variables

You can configure `nshutils` using environment variables:

```bash
# Enable debug mode
export NSHUTILS_DEBUG=1

# Enable typecheck (or disable with 0)
export NSHUTILS_TYPECHECK=1

# Enable ActSave with default temp directory
export NSHUTILS_ACTSAVE=1

# Enable ActSave with specific directory
export NSHUTILS_ACTSAVE="/path/to/activations"

# Set ActSave filters (comma-separated patterns)
export NSHUTILS_ACTSAVE_FILTERS="layer*,attention*,encoder.*"

# JSON configuration (overrides individual variables)
export NSHUTILS_CONFIG='{"debug": {"enabled": true}, "typecheck": {"enabled": false}, "actsave": {"enabled": true, "save_dir": "/path/to/activations", "filters": ["layer*"]}}'

# Comma-separated configuration
export NSHUTILS_CONFIG="debug=true,typecheck=false,actsave=true"
```

### Programmatic Configuration

You can also configure `nshutils` programmatically:

```python
from nshutils import config

# Enable debug mode
config.set(True, "debug")

# Configure typecheck with explicit settings
config.set({"enabled": True}, "typecheck")

# Configure ActSave
config.set({
    "enabled": True,
    "save_dir": "/path/to/activations",
    "filters": ["layer*", "attention*"]
}, "actsave")

# Check current settings
print(f"Debug enabled: {config.debug_enabled()}")
print(f"Typecheck enabled: {config.typecheck_enabled()}")
print(f"ActSave enabled: {config.actsave_enabled()}")

# Temporary overrides using context managers
with config.debug_override(False):
    # Debug is temporarily disabled
    pass

with config.actsave_override({"enabled": True, "filters": ["encoder.*"]}):
    # ActSave temporarily uses different filters
    pass
```

### Configuration Hierarchy

The configuration system follows a hierarchical approach:

1. **Debug → Typecheck**: When debug is enabled, typecheck is automatically enabled unless explicitly overridden
2. **Environment variables** take precedence over programmatic settings
3. **Context managers** provide temporary overrides within their scope

## Features

### Typechecking

`nshutils` provides a simple way to typecheck your code using the [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping) library. The typecheck system is now integrated with the unified configuration system.

Enable typechecking globally:

```bash
export NSHUTILS_TYPECHECK=1
```

Or programmatically:

```python
from nshutils import config
config.set(True, "typecheck")

# Now use typecheck decorators
from nshutils.typecheck import typecheck

@typecheck
def my_function(x: Float[torch.Tensor, "batch seq"]) -> Float[torch.Tensor, "batch seq"]:
    return x * 2
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

`nshutils` provides a simple way to save and load activations from neural networks. ActSave is now integrated with the unified configuration system.

#### Basic Usage

Enable ActSave via environment variables:

```bash
# Enable with default temp directory
export NSHUTILS_ACTSAVE=1

# Enable with specific directory
export NSHUTILS_ACTSAVE="/path/to/activations"

# Set filters (comma-separated patterns)
export NSHUTILS_ACTSAVE_FILTERS="layer*,attention*"
```

Or programmatically:

```python
from nshutils import config
from nshutils import ActSave

# Enable ActSave with configuration
config.set({
    "enabled": True,
    "save_dir": "/path/to/activations",
    "filters": ["layer*", "attention*"]
}, "actsave")

def my_model_forward(x):
    ...
    # Save activations - automatically filtered
    ActSave({"encoder.activations": x})

    # Equivalent to the above
    with ActSave.context("encoder"):
        ActSave(activations=x)
    ...

x = torch.randn(...)
my_model_forward(x)
# Activations are saved to disk under the configured directory
```

You can also use the traditional explicit enable/disable pattern:

```python
from nshutils import ActSave

# Explicit enable with filters
ActSave.enable(save_dir="path/to/activations", filters=["layer*", "attention*"])

def my_model_forward(x):
    ActSave({"encoder.activations": x})

x = torch.randn(...)
my_model_forward(x)
ActSave.disable()
```

#### Activation Filtering

ActSave supports filtering to selectively save only certain activations based on fnmatch patterns. This is useful for reducing storage space and focusing on specific model components.

Configure filters via environment variables:

```bash
# Only save activations matching "layer*" or "attention*" patterns
export NSHUTILS_ACTSAVE_FILTERS="layer*,attention*"
export NSHUTILS_ACTSAVE=1
```

Or programmatically:

```python
from nshutils import config

# Configure via unified config system
config.set({
    "enabled": True,
    "save_dir": "/path/to/activations",
    "filters": ["layer*", "attention*"]
}, "actsave")

# These will be saved (match filters)
ActSave(
    layer1_output=x1,
    layer2_hidden=x2,
    attention_weights=x3
)

# These will NOT be saved (don't match filters)
ActSave(
    decoder_output=x4,
    embedding_vector=x5
)
```

Traditional explicit filtering is also supported:

```python
from nshutils import ActSave

# Only save activations matching "layer*" or "attention*" patterns
filters = ["layer*", "attention*"]

with ActSave.enabled(save_dir="path/to/activations", filters=filters):
    # These will be saved (match filters)
    ActSave(
        layer1_output=x1,
        layer2_hidden=x2,
        attention_weights=x3
    )

    # These will NOT be saved (don't match filters)
    ActSave(
        decoder_output=x4,
        embedding_vector=x5
    )
```

The filtering patterns support standard Unix shell-style wildcards:

- `*` matches everything
- `?` matches any single character
- `[seq]` matches any character in seq
- `[!seq]` matches any character not in seq

##### Contextual Filtering

Filters work with context prefixes, allowing you to save activations from specific model components:

```python
from nshutils import config, ActSave

# Only save activations from encoder layers
config.set({
    "enabled": True,
    "save_dir": "/path/to/activations",
    "filters": ["encoder.*"]
}, "actsave")

# Decoder context - these won't be saved
with ActSave.context("decoder"):
    ActSave(layer1_output=x1, attention=x2)

# Encoder context - these will be saved
with ActSave.context("encoder"):
    ActSave(layer1_output=x3, attention=x4)  # Saved as "encoder.layer1_output", "encoder.attention"
```

##### Dynamic Filter Updates

You can update filters during runtime:

```python
from nshutils import config

# Set initial filters
config.set({"enabled": True, "filters": ["layer*"]}, "actsave")

# Initially only layer outputs saved
ActSave(layer1_output=x1, attention_weights=x2)

# Update filters
config.set({"enabled": True, "filters": ["attention*"]}, "actsave")
ActSave(layer2_output=x3, attention_weights=x4)  # Only attention_weights saved

# Check current filters
current_filters = config.actsave_filters()  # Returns ["attention*"]

# Clear filters (save all)
config.set({"enabled": True, "filters": None}, "actsave")
```

Traditional explicit filter management is also available:

```python
ActSave.enable(save_dir="path/to/activations")

# Initially no filters - all activations saved
ActSave(layer1_output=x1, attention_weights=x2)

# Update to only save layer outputs
ActSave.set_filters(["layer*"])
ActSave(layer2_output=x3, decoder_output=x4)  # Only layer2_output saved

# Check current filters
current_filters = ActSave.filters  # Returns ["layer*"]

# Clear filters
ActSave.set_filters(None)
```

##### Environment Variable Configuration

You can configure ActSave and filtering through environment variables (now part of the unified configuration system):

```bash
# Enable ActSave with default temp directory
export NSHUTILS_ACTSAVE=1

# Enable ActSave with specific directory
export NSHUTILS_ACTSAVE="/path/to/activations"

# Set filters (comma-separated patterns)
export NSHUTILS_ACTSAVE_FILTERS="layer*,attention*,encoder.*"

# Combine both
export NSHUTILS_ACTSAVE="/path/to/activations"
export NSHUTILS_ACTSAVE_FILTERS="layer*,attention*"

# Or use the unified config format
export NSHUTILS_CONFIG='{"actsave": {"enabled": true, "save_dir": "/path/to/activations", "filters": ["layer*", "attention*"]}}'
```

The `NSHUTILS_ACTSAVE_FILTERS` environment variable supports:

- **Comma-separated patterns**: `"layer*,attention*,decoder.*"`
- **Whitespace handling**: Extra spaces around commas are automatically trimmed
- **Empty values**: Empty string or only commas/spaces result in no filtering

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
