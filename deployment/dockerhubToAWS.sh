docker pull nginx
docker tag nginx:latest <USER_NAME>/<REPO_NAME>:<TAG_NAME>
docker push <USER_NAME>/<REPO_NAME>:<TAG_NAME>
aws kms create-key --query KeyMetadata.Arn --output text
aws kms create-alias --alias-name alias/<CMK_ALIAS> --target-key-id <CMK_ARN>
aws secretsmanager create-secret \
--name dev/DockerHubSecret \
--description "Docker Hub Secret" \
--kms-key-id alias/<CMK_ALIAS> \
--secret-string '{"username":"<USER_NAME>","password":"<PASSWORD>"}'

cat << EOF > ecs-trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

cat << EOF > ecs-secret-permission.json 
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "secretsmanager:GetSecretValue"
      ],
      "Resource": [
        "<SECRET_ARN>",
        "<CMK_ARN>"     
      ]
    }
  ]
}
EOF

aws iam create-role \
--role-name ecsTaskExecutionRole \
--assume-role-policy-document file://ecs-trust-policy.json

aws iam attach-role-policy \
--role-name ecsTaskExecutionRole \
--policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam put-role-policy \
--role-name ecsTaskExecutionRole \
--policy-name ECS-SecretsManager-Permission \
--policy-document file://ecs-secret-permission.json

ecs-cli configure profile \
--access-key <AWS_ACCESS_KEY_ID> \
--secret-key <AWS_SECRET_ACCESS_KEY> \
—profile-name <PROFILE_NAME>

ecs-cli configure \
--cluster <CLUSTER_NAME> \
--default-launch-type <LAUNCH_TYPE> \
--config-name <CONFIG_NAME> \
--region <AWS_REGION>

ecs-cli up \
--cluster <CLUSTER_NAME> \
--region us-east-1 \
--launch-type FARGATE \

aws ec2 describe-security-groups \
--filters Name=vpc-id,Values=<VPC_ID> \
--region us-east-1

aws ec2 authorize-security-group-ingress \
--group-id <SG_ID> \
--protocol tcp \
--port 80 \
--cidr 0.0.0.0/0 \
--region us-east-1

# begin service
cat << EOF > docker-compose.yml
version: "3"
services:
    web:
        image: <USER_NAME>/<REPO_NAME>:<TAG_NAME>
        ports:
            - 80:80
EOF

cat << EOF > ecs-params.yml
version: 1
task_definition:
  task_execution_role: ecsTaskExecutionRole
  ecs_network_mode: awsvpc
  task_size:
    mem_limit: 0.5GB
    cpu_limit: 256
  services:
    web:
        repository_credentials: 
            credentials_parameter: "<SECRET_ARN>"
run_params:
  network_configuration:
    awsvpc_configuration:
      subnets:
        - "<SUB_1_ID>"
        - "<SUB_2_ID>"
      security_groups:
        - "<SG_ID>"
      assign_public_ip: ENABLED
EOF

ecs-cli compose \
--project-name <PROJECT_NAME> \
--cluster <CLUSTER_NAME> \
service up \
--launch-type FARGATE

ecs-cli compose \
--project-name <PROJECT_NAME> \
--cluster <CLUSTER_NAME> \
service ps

