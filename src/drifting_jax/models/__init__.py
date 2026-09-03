"""JAX model definitions.

Import concrete models from their submodules.  Keeping this initializer free of
eager imports prevents a cycle between model definitions and checkpoint
initialization.
"""
