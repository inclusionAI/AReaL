#!/usr/bin/env bash

set -e

GIT_REPO_URL=${GIT_REPO_URL:?"GIT_REPO_URL is not set"}
GIT_COMMIT_SHA=${GIT_COMMIT_SHA:?"GIT_COMMIT_SHA is not set"}

echo "GIT_REPO_URL: $GIT_REPO_URL"
echo "GIT_COMMIT_SHA: $GIT_COMMIT_SHA"

RUN_ID="areal-$GIT_COMMIT_SHA"
rm -rf "/tmp/$RUN_ID"
mkdir -p "/tmp/$RUN_ID"
cd "/tmp/$RUN_ID"

git init
git remote add origin "$GIT_REPO_URL"
git fetch --depth 1 origin "$GIT_COMMIT_SHA"
git checkout FETCH_HEAD
