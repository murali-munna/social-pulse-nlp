aws ecr create-repository --repository-name [repository-name] --region [region]
docker tag [name] aws_account_id.dkr.ecr.region.amazonaws.com/[repository-name]
aws ecr get-login-password | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com
docker push aws_account_id.dkr.ecr.region.amazonaws.com/hello-repository
# clear
aws ecr delete-repository --repository-name [repository-name]  --region [region] --force
