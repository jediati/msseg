"""Compatibility shim: msseg.mscoupon.torch_mlp moved to msseg.labeler.torch_mlp.

Kept so existing imports keep working and -- for the classifier pickles, which
record a step's class by module path -- so msseg.mscoupon.torch_mlp.<Class>
still resolves to the very same class object. Every public and private name
of the framework module is re-exported; new code should import the framework
module directly.
"""
from msseg.labeler import torch_mlp as _framework_module

globals().update({k: v for k, v in vars(_framework_module).items()
                  if not k.startswith("__")})
del _framework_module
