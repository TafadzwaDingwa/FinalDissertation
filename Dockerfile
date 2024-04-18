FROM python:3.8-slim-buster 

WORKDIR /app

# Copy model and script
COPY my_model.h5 /app/my_model.h5  
COPY my_model.py /app/my_model.py  
COPY Benign_vs_DDoS.csv /app/data/Benign_vs_DDoS.csv

# Create requirements.txt within the container
RUN pip3 freeze > requirements.txt  # Generate requirements dynamically

# Install dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install tensorflow
RUN pip3 install numpy 
RUN pip3 install scikit-learn 
RUN pip3 install pandas


# Expose port for predictions
EXPOSE 3000

# Set environment variables (if needed)
ENV MODEL_NAME="MyDDoSDetector"  

# Define start command
CMD ["python", "/app/my_model.py"]