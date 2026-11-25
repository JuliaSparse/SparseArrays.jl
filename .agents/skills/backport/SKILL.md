---
name: backport
description: Assemble a backport pull request onto a release branch - choosing the PRs, cherry-picking them in order, resolving conflicts, the checklist body, the release history, and checking dependent packages. Use when asked to backport merged PRs to a Julia release.
---

# Backports

Each Julia release ships its own copy of this package, from a release branch. Changes
reach it only through a backport PR into that branch, and reach users only after Julia's
own release branch bumps to the new commit.

## Choosing what goes in

- The maintainers decide by labelling the merged PRs with the release's backport label.
  Never add or remove a label yourself; saying in a PR that it is a backport candidate is
  fine.
- Before the release's `v1.x.0` ships, its branch takes every labelled PR. Afterwards it
  takes only bug and regression fixes: no new methods, no behaviour changes, nothing that
  changes a result that working code relies on.
- Never bump compat on a release branch, and never push to one: everything goes through
  the backport PR.

## Building the branch

1. Branch from the current release branch, in its own worktree.
2. Cherry-pick each labelled PR's squashed commit with `git cherry-pick -x`, in the order
   the PRs merged on `main`, so that later picks apply on top of the code they were
   written against.
3. On a conflict, find the PR on `main` that the pick depends on. Either include that PR
   and say why, or adapt the pick to the release branch's code and say how. If neither is
   reasonable, leave the PR out and say so; never resolve a conflict by guessing.
4. Run the tests on the release's Julia, not nightly; CI on the release branch does too.
   A test that the pick adds may need a fix that only `main` has; say so if you include
   one.
5. Move the release-history entries of every included PR from Unreleased into the
   release's section, unless that release has already shipped (see the
   `release-history` skill).

When the release has not shipped and every PR since the branch point is included, the
branch's tree may equal `main` at some commit. Check with `git diff` and say so; it is the
strongest statement of what the branch contains.

## The pull request

- Title: "Backports for release 1.x". Head branch: the conventional backports branch for
  that release.
- Body: one sentence on how the picks were made (labelled PRs, `-x`, merge order), then a
  checklist of the PR numbers in pick order, each ticked once it is in and CI passed.
- Then every deviation: a PR picked out of order, a labelled PR left out and why, an
  unlabelled PR included and why, a pick adapted to the branch, and which included PRs
  are not bug fixes. A reviewer should not need to compare the list with the labels.
- For a large backport, add PRs one push at a time and check CI before the next, so that
  a failure points at one pick.

## Checking dependent packages

A large backport, or one that changes results, gets a downstream check before it merges:

- Run the test suites of the packages that depend directly on SparseArrays twice, on the
  release's Julia with its bundled copy of the package and with the branch in its place,
  in separate depots from one frozen registry snapshot. Confirm which copy each run
  loaded.
- Choose the most-used direct dependents, a few per organisation, plus every package of
  the sparse-matrix organisation. Skip archived packages and ones whose tests need
  Python or run for hours.
- Post the comparison as a table comment on the backport PR and edit it in place as
  results arrive. For each failure, show whether it also fails with the bundled copy.

## After it merges

Say in the PR, or in a comment once it merges, that Julia's release branch needs a
SparseArrays bump to pick the backports up.
