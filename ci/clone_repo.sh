#!/usr/bin/env bash

set -e

GIT_REPO=${GIT_REPO:?"GIT_REPO is not set"}
GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}

echo "GIT_REPO: $GIT_REPO"
echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

RUN_ID="areal-$GIT_COMMIT_SHA"
mkdir -p "/tmp/$RUN_ID"
cd "/tmp/$RUN_ID"

git init
git remote add origin "https://github.bibk.top/$GIT_REPO"
git fetch --depth 1 origin "$GIT_COMMIT_SHA"
git checkout FETCH_HEAD
