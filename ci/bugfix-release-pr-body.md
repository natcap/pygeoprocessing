AUTO: merge $RELEASE_BRANCH into $SOURCE_BRANCH

This PR was automatically generated in response to a push to `master`,
and is a chance to review any changes that will be included in the release
branch before merging.  Under most circumstances, this PR will probably be
a formality.  However, there are a few cases where we may need to be some
extra work to make sure `$RELEASE_BRANCH` contains what it should after the
merge:

## If this PR causes a trivial merge conflict

Use the github merge conflict resolution editor to resolve the change and
commit the change to a new branch, *not to master*.

## If this PR causes a nontrivial merge conflict

1. Decline this PR
2. Make a new bugfix branch off of `master`
3. Merge `$RELEASE_BRANCH` into the bugfix branch, resolving the conflict.
4. PR the bugfix branch into `$RELEASE_BRANCH`

## If this PR contains content that should not be in $RELEASE_BRANCH

1. Decline this PR
2. Make a new bugfix branch off of `master`
3. Merge `$RELEASE_BRANCH` into the bugfix branch
4. Handle the content that should not end up in `$RELEASE_BRANCH` however it
   needs to be handled
5. Commit the updated content
6. PR the bugfix branch into `$RELEASE_BRANCH`

## What happens if we accidentally merge something we shouldn't?

There are several possibilities for recovery if we get to such a state.

1. A merge can be undone through the github interface if the error is caught
   directly after the PR is merged.
2. If we're commits in past the erroneous merge, create a branch off of
   `$RELEASE_BRANCH`, back out of the changes or edit files needed to resolve
   the issue, and PR the branch back into `$RELEASE_BRANCH`.

### Why was this PR created?

Possible events that can trigger this include:

* Other pull requests for feature or bugfixes being merged into `master`
* Automated bugfix releases
* Any manual push to `master`, if ever that happens (which shouldn't be the
  case given our branch protections)

The workflow defining this PR is located at
`.github/workflows/auto-pr-from-master-into-releases.yml`.
