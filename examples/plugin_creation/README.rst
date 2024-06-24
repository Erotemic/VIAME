
========================
External Plugin Creation
========================

This document corresponds to `this example online`_, in addition to the external_plugin_creation
example folder in a VIAME installation.

.. _this example online: https://github.com/VIAME/VIAME/tree/master/examples/external_plugin_creation

This directory contains the source files needed to make a loadable algorithm plugin implementation
external to VIAME, which links against an installation, or in the case of python generates a loadable
script. This is for cases where we might want to just make a plugin against pre-compiled binaries,
instead of building all of VIAME itself.

The procedure is slightly different depending on whether you are developing an external C++ or
Python module. C++ modules require linking your module against VIAME, the output of which is 
plugin DLL which can be used directly in VIAME pipelines. Python processes can be made without
compilation, and placed in your PYTHONPATH for use by the plugin system.
