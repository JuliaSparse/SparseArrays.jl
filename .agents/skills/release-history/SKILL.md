---
name: release-history
description: Add, move and maintain entries in the release history - which changes need one, how to write it, which release it goes under, and what happens to the entries when a change is backported or a release branch is cut. Use for any PR that changes what users see, and when preparing a backport or a release.
---

# The release history

The release history lists, for each Julia release, the changes users of this package can
notice. Its header states what is listed; read it before writing an entry. An entry
written later, from a squashed commit, loses what the PR author knew, so every PR that
needs one adds it.

## Which changes need an entry

- **New**: a public function, method, keyword or type that was not there, or an
  operation that threw and now works.
- **Changed**: a result that code could have relied on is different: its type, its
  value, its stored zeros, its index type, an error thrown instead of a result. A bug fix
  counts only when it changes such a result, not when it turns a crash into the
  documented answer.
- **Faster**: a measured improvement users would notice, including allocation, compile
  and load time.
- **Deprecated** and **Removed**.

No entry for internal helpers, experimental features, tests, CI, documentation, or a
restructuring that changes no result.

## Writing an entry

- One bullet: the mark in bold, then a sentence in the user's terms, naming the public
  function and the case, and what it does now, with the old behaviour when that is what
  a reader needs to update their code. Not the mechanism: "visits only the stored
  entries", not the name of a kernel.
- End it with the PR reference, and add the reference's link definition at the bottom
  of the file, in numerical order. The number is known once the PR is open; add the
  entry in a commit after opening it.
- Several PRs that make one user-visible change share one entry with all their
  references. Extend an existing entry rather than adding a second one about the same
  function.
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
