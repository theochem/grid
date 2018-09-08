#!/usr/bin/env python
"""Get the version string from the git tag."""

from __future__ import print_function

import subprocess


__all__ = ['get_gitversion']


TEMPLATES = {
    'python': """\
\"""Do not edit this file because it will be overwritten before packaging.

The output of git describe was: {git_describe}
\"""
__version__ = '{git_tag_version}'""",
    'cmake': """\
# This file is automatically generated. Changes will be overwritten before packaging.
set(GIT_DESCRIBE "{git_describe}")
set(GIT_TAG_VERSION "{git_tag_version}")
set(GIT_TAG_SOVERSION "{git_tag_soversion}")
set(GIT_TAG_VERSION_MAJOR "{git_tag_version_major}")
set(GIT_TAG_VERSION_MINOR "{git_tag_version_minor}")
set(GIT_TAG_VERSION_PATCH "{git_tag_version_patch}")"""}


def get_gitversion():
    """Return a conda-compatible version string derived from git describe --tags."""
    git_describe = subprocess.check_output(['git', 'describe', '--tags']).strip()
    version_words = git_describe.decode('utf-8').strip().split('-')
    version = version_words[0]
    if len(version_words) > 1:
        version += '.post' + version_words[1]
    return version, git_describe


def main():
    """Print the version derived from ``git describe --tags`` in a useful format."""
    from argparse import ArgumentParser
    parser = ArgumentParser('Determine version string from `git describe --tags`')
    parser.add_argument('output', choices=['plain', 'python', 'cmake'], default='plain', nargs='?',
                        help='format of the output.')
    args = parser.parse_args()
    version, git_describe = get_gitversion()
    if args.output == 'plain':
        print(version)
    else:
        major, minor, patch = version.split('.', 2)
        print(TEMPLATES[args.output].format(
            git_describe=git_describe,
            git_tag_version=version,
            git_tag_soversion='.'.join([major, minor]),
            git_tag_version_major=major,
            git_tag_version_minor=minor,
            git_tag_version_patch=patch,
        ))


if __name__ == '__main__':
    main()
