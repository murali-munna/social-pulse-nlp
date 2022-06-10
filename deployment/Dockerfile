# base 
FROM ubuntu:20.04

# install python3
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update &&\
    apt-get install python3.7 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y


# Expose port 8501
EXPOSE 8501

# Working Directory
WORKDIR /app

#ARG DEBIAN_FRONTEND=noninteractive
#ENV DEBIAN_FRONTEND=noninteractive 


# Copy source code to working directory
COPY requirements.txt ./requirements.txt

# Install packages from requirements
RUN pip3 install Cython
RUN pip3 install -r requirements.txt

# Copy source code to working directory
COPY . /app/

# Run app.py at container launch
CMD streamlit run app.py

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
