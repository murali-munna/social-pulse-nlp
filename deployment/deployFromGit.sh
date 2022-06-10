git clone https://github.com/aws-samples/amazon-ecs-cli-sample-app.git demo-app && \ 
cd demo-app &&                               \
copilot init --app demo                      \
  --name api                                 \
  --type 'Load Balanced Web Service'         \
  --dockerfile './Dockerfile'                \
  --port 80                                  \
  --deploy

# check app
copilot app ls
# chech service
copilot app show
# check env
copilot env ls
# show resource
copilot svc show
# check service list
copilot svc ls
# check log
copilot svc logs
# check status
copilot svc status
# delete
copilot app delete

