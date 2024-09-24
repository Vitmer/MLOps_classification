# MLOps Project

## Overview
This project is an API for product classification using text and image data. The API is built with FastAPI and uses a machine learning model for predictions.

## Project Structure
- `src/api/`: Contains the main API code.
- `src/models/`: Contains the saved machine learning model and vectorizer.
- `src/data/`: Contains raw and processed data.
- `src/tests/`: Contains unit tests for the API, model.
- `Dockerfile`: Used for containerizing the application.
- `docker-compose.yml`: Used for orchestrating the application services.

## Getting Started
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the application:
    ```bash
    uvicorn src.api.app:app --reload
    ```
3. Run tests:
    ```bash
    pytest src/tests/
    ```
4. Build Docker container:
    ```bash
    docker-compose up --build
    ```

    1. Application and Database Separation on Infrastructure Level

	•	Application (API and Machine Learning):
	•	The application is deployed in a Docker container on EC2. This ensures flexibility in updates and scaling.
	•	For automated deployment, use GitHub Actions for CI/CD. Each change in the repository triggers the build process and updates the container on EC2.
	•	Docker Compose can be used to organize services (e.g., API, monitoring, database) within a single container.
	•	Database:
	•	For the database, I recommend using Amazon RDS. It’s a specialized platform for data storage that handles scalability, fault tolerance, and backups.
	•	Using RDS allows the separation of the database from the application, making it easier to scale in the future.
	•	S3 can be used to store large files (e.g., models or data) if they are not suitable for RDS.

2. CI/CD with GitHub Actions

	•	Set up GitHub Actions for automatic Docker image builds when changes are made to the repository and deploy the updated container on EC2.
	•	In GitHub Actions, steps are defined for pushing Docker images to Amazon Elastic Container Registry (ECR) and automatically updating EC2.
	•	If the code is updated, the model and API are automatically tested and then deployed.

3. Monitoring and Automation (MLOps)

	•	Use Prometheus and Grafana to monitor API health and model performance (including metrics like F1-score).
	•	Auto-scaling for EC2 via Amazon Auto Scaling for automatic resource scaling depending on the load on the API.
	•	Auto-retraining based on threshold values of F1-score or other metrics. If the metrics drop in Grafana or Prometheus, the model retraining process is automatically triggered.

4. Security and User Management

	•	IAM roles and policies for managing access to EC2, S3, RDS, and other AWS services.
	•	OAuth2 or AWS Cognito for secure API user authentication.
	•	VPC (Virtual Private Cloud) for secure connections between your API (EC2) and the database (RDS).

Architecture:

	1.	EC2 (Docker with API and Monitoring):
	•	FastAPI application
	•	Docker image for the application
	•	Monitoring via Grafana/Prometheus
	2.	RDS for Database:
	•	Manage and store data using a specialized solution for relational databases.
	3.	S3 for Model and Large File Storage:
	•	Store machine learning models, large datasets, or database backups.
	4.	CI/CD with GitHub Actions:
	•	
	Automate the build, testing, and deployment of containers.
	
	

	