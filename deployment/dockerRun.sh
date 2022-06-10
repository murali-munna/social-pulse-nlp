docker image build -t streamlit_test:app
docker image ls [OPTIONS] [REPOSITORY[:TAG]]
docker container run -p 8501:8501 streamlit_test:app
