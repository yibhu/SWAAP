"""Pinned family implementations, loaded explicitly by swaap.latent.families.

The former broad, exception-swallowing dispatcher is not required: installation
errors for one requested family now propagate rather than importing every family.
Numerical environment wrapper methods are retained unchanged.
"""
