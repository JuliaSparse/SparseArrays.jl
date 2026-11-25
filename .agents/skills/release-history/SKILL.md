---
name: release-history
description: Add, move and maintain entries in the release history - which changes need one, how to write it, which release it goes under, and what happens to the entries when a change is backported or a release branch is cut. Use for any PR that changes what users see, and when preparing a backport or a release.
---

# The release history

The release history tells a user what a release changes for them, at a high level. It is
strategic, not a list of PRs: a reader scanning a release should see its few themes and
anything that makes them update their code. Its header states what is listed; read it
before writing an entry. An entry written later, from a squashed commit, loses what the
PR author knew, so every PR that needs one adds it or extends an existing one.

## Keep it high level

- One entry per theme, not per PR or per function. A series of PRs on one area (say,
  faster products, or one set of rules for every route of an operation) is one entry
  with all their references.
- Name the area and the effect, not each function, wrapper and case: "Sparse products
  are faster, most of all `A*B`", not a list of the forms that changed.
- Prefer extending an existing entry, by its wording or just its references, to adding a
  new one.
- Small fixes, edge cases, and fixes of a regression that no release shipped get no
  entry.
- What breaks existing code always gets one, in the release's upgrading section when it
  has one.

## Which changes need an entry

- **New**: a public function, method, keyword or type that was not there, or an
  operation that threw and now works.
- **Changed**: a result that code could have relied on is different: its type, its
  value, its stored zeros, its index type, an error thrown instead of a result. A bug fix
  counts only when it changes such a result broadly enough to matter, not when it turns a
  crash into the documented answer.
- **Faster**: a measured improvement users would notice in common use, including
  allocation, compile and load time.
- **Deprecated** and **Removed**.

No entry for internal helpers, experimental features, tests, CI, documentation, or a
restructuring that changes no result.

## Writing an entry

- One bullet: the mark in bold, then one or two sentences in the user's terms on what
  now behaves differently, with the old behaviour when a reader needs it to update their
  code. Not the mechanism, and never the name of a kernel.
- End it with the PR reference, and add the reference's link definition at the bottom
  of the file, in numerical order. The number is known once the PR is open; add the
  entry in a commit after opening it.
- Within a list, entries go in the order of the marks: New, Changed, Faster, Deprecated,
  Removed.

## Which release it goes under

A change goes under the first Julia `v1.x.0` release that ships it.

- A change merged on `main` goes under **Unreleased**.
- When it is backported to a release branch whose `v1.x.0` has not shipped, its entry
  moves from Unreleased to that release's section, into the topic group it belongs to.
  The backport PR makes that move for every PR it contains.
- A fix backported to an already shipped release (a patch release) stays where it is:
  patch releases are not listed.
- A change that alters code written for the previous release also goes in that
  release's upgrading section, with a before-and-after example when one makes it clear.

## Cutting a release

When a release branch is cut from `main`, Unreleased becomes that release's section:
give it the line on which Julia ships it and the SuiteSparse version, group the entries
by topic if there are many, and add highlights for a large release. Start a new empty
Unreleased section. When the release ships, replace "not yet released" with the date.

## Checking

- Every reference has a link definition and every definition is used.
- Nothing is listed twice, and no backported change remains under Unreleased.
