dockerpath="user/SOCIAL-PULSE-NLP"

# Authenticate & Tag
echo "Docker ID and Image: $dockerpath"
docker login &&\
    docker image tag SOCIAL-PULSE-NLP $dockerpath

# Push Image
docker image push $dockerpath