# syntax = devthefuture/dockerfile-x
# SPDX-License-Identifier: MIT

# Syntax provided by the devthefuture/dockerfile-x project
# Will copy in the base configuration for the build
INCLUDE ./docker/common/base.dockerfile

# Snippets
# INCLUDE ./docker/snippets/miniconda.dockerfile
INCLUDE ./docker/snippets/libsurvive.dockerfile
INCLUDE ./docker/snippets/librealsense2.dockerfile

# Will copy in other common configurations for this build
INCLUDE ./docker/common/common.dockerfile

# Complete the build
INCLUDE ./docker/common/final.dockerfile
