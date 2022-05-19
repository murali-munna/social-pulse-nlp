
FROM python:3.7.3-stretch

# Working Directory
WORKDIR /app
# Copy source code to working directory
COPY . /app/

# Install packages from requirements.txt
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirement.txt

# Expose port 8080
EXPOSE 8080

# Run app.py at container launch
CMD ["python", "apps/app2.py"]