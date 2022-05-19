#!/usr/bin/env bash
# Build image
/usr/local/bin/docker build --tag = SOCIAL-PULSE-NLP .
# List docker images
/usr/local/bin/docker image ls
# Run flask app
docker run -p 8080:8080 SOCIAL-PULSE-NLP