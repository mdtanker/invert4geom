# Release process for `invert4geom`

We use *semantic versioning*, where version numbers are classified
as v<major>.<minor>.<patch>.  By default, releases are made from the main
branch as part of a linear release history and, as described below, are
triggered by pushing a git tag to the invert4geom repository on github.  If
a patch release is required for an older version, a branch can be created from
the appropriate point in main and the following instructions are still apt.

## Update the release notes:

  1. Make a list of merges, contributors, and reviewers::

    ### Set release variables:

      export VERSION=<about-to-be-released version number>
      export PREVIOUS=<previous version number>

    ### Autogenerate release notes

      changelist mdtanker/invert4geom v${PREVIOUS} main --version ${VERSION} --config pyproject.toml --out ${VERSION}.md

    ### Put the output of the above command at the top of `CHANGELOG.md`

      cat ${VERSION}.md | cat - CHANGELOG.md > temp && mv temp CHANGELOG.md


  2. Scan the PR titles for highlights and mention these in the
     relevant sections of the notes.
     Ideally, all changed API objects are mentioned by name,
     e.g. a new parameter or a deprecated function.
