# Invert4geom
3D geometric gravity inversions

```{include} ../README.md
:start-after: <!-- SPHINX-START1 -->
:end-before: <!-- SPHINX-END1 -->
```

```{image} figures/cover_fig.png
:alt: cover figure
:width: 400px
:align: center
```

```{include} ../README.md
:start-after: <!-- SPHINX-START2 -->
:end-before: <!-- SPHINX-END2 -->
```

```{toctree}
:maxdepth: 1
:hidden:
overview
install
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 🚶 User guide
user_guide/simple_inversion
user_guide/damping_cross_validation
user_guide/density_cross_validation
user_guide/reference_level_cross_validation
user_guide/including_starting_model
user_guide/adhering_to_constraints
user_guide/combining_it_all
user_guide/estimating_regional_field
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: 📚 Gallery
gallery/index.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📖 Reference documentation
api/invert4geom
citing.md
changelog.md
references.rst
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: ℹ️ Other resources
contributing.md
Source code on GitHub <https://github.com/mdtanker/invert4geom>
```


::::{grid} 2
:::{grid-item-card} {octicon}`rocket` Getting started?
:text-align: center
New to Invert4geom? Start here!
```{button-ref} overview
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::

:::{grid-item-card} {octicon}`comment-discussion` Need help?
:text-align: center
Start a discussion on GitHub!
```{button-link} https://github.com/mdtanker/invert4geom/discussions
    :click-parent:
    :color: primary
    :outline:
    :expand:
    Discussions
```
:::

:::{grid-item-card} {octicon}`file-badge` Reference documentation
:text-align: center
A list of modules and functions
```{button-ref} api/invert4geom
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::

:::{grid-item-card} {octicon}`bookmark` Using Invert4geom for research?
:text-align: center
Citations help support our work
```{button-ref} citing
    :click-parent:
    :color: primary
    :outline:
    :expand:
```
:::
::::


```{admonition} Early-stages of development
:class: seealso

This package is at the very beginning of it's development! This means that we are still adding a lot of new features and sometimes we
make changes to the ones we already have while we try to improve the
software based on users' experience, test new ideas, take better design
decisions, etc.
Some of these changes could be **backwards incompatible**. Keep that in
mind before you update Invert4Geom to a newer version.
```

```{admonition} How to contribute
:class: seealso

Please, read our [Contributor Guide](https://github.com/mdtanker/invert4geom/blob/main/.github/CONTRIBUTING.md) to learn
how you can contribute to the project.
```

```{note}
*Many parts of this documentation was adapted from the* [Fatiando project](https://www.fatiando.org/).
```
