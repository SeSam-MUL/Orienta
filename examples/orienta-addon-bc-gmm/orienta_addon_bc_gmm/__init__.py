"""The reference Orienta add-on. See ``analysis.py`` and the README.

Deliberately empty of imports. Orienta puts this package's parent directory on
``sys.path`` and then imports ``orienta_addon_bc_gmm.analysis``; anything done
here would run before the analysis does, for no benefit and at the cost of
making the import surface larger than the one line the manifest names.
"""
