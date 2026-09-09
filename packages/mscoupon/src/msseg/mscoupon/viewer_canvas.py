"""Compatibility shim: msseg.mscoupon.viewer_canvas moved to msseg.labeler.canvas.

Kept so existing imports keep working and -- for the classifier pickles, which
record a step's class by module path -- so msseg.mscoupon.viewer_canvas.<Class>
still resolves to the very same class object. Every public and private name
of the framework module is re-exported; new code should import the framework
module directly.
"""
from msseg.labeler import canvas as _framework_module

globals().update({k: v for k, v in vars(_framework_module).items()
                  if not k.startswith("__")})
del _framework_module
