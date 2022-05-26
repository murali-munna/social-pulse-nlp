# base 
FROM ubuntu:18.04

# install python3
RUN apt-get update &&\
    apt-get install python3.7 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y

# Expose port 8080
EXPOSE 8080

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY requirements.txt ./requirements.txt

# Install packages from requirements.txt
RUN pip install --upgrade pip &&\
    pip3 install -r requirements.txt

# Copy source code to working directory
COPY . /app/

# Run app.py at container launch
CMD streamlit run streamlit_app/demo.p

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'