# Dependencies only: the immutable runner executable is supplied by Arena.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
RUN python3 -m pip install --no-cache-dir hermes-agent==0.19.0 math-verify==0.8.0
# The selected base image must supply bubblewrap and permit rootless namespaces.
RUN command -v bwrap
USER 1000
