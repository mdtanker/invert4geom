# Overview

Invert4geom provides a series of tools for conducting a specific style of
gravity inversion. Many gravity inversions aim to model the density distribution
of the subsurface. These are commonly used to identify bodies of anomalous
densities, such as igneous intrusions or ore deposits. The typical way these are
performed is to _discretize_ the subsurface into a series of finite volumes,
such as cubes or prisms, where the shape of the volumes doesn't change. The
inversion then alters the density values of each of these volumes to match the
observed gravity anomaly. In these inversions the _density_ values changes,
while the _geometry_ of the volumes remains unchanged. These types of inversions
may be referred to as _density inversions_. Here, instead, we are performing
_geometric inversions_.

Geometric inversions are essentially the opposite. The density values of the
volumes in the discretized model remain unchanged, while their geometry is
altered. Here we use layers of vertical right-rectangular prisms and alter their
_tops_ and _bottoms_ during the inversion. Typically use cases for these style
of inversion are modeling the topography of the Moho, the contact between
sediment and basement, or the shape of the seafloor in locations where it is not
easily mapped.

Currently, this package is only intended to perform inversions using right
rectangular prisms. Other types of volumes, such as tesseroids, are currently
not implemented.

Much of this software was developed as part of my Ph.D. thesis. For detailed
description of the theory and implementation of this inversion, as well as many
synthetic tests and a real-world application to modelling bathymetry, see
chapter 3 and 4 of my thesis, available
[here](https://doi.org/10.26686/wgtn.24408304). The code was originally included
in [this GitHub repository](https://github.com/mdtanker/RIS_gravity_inversion),
but much of it has been migrated here.
