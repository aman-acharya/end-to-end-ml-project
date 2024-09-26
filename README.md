# End-to-End Machine Learning Project

## Project Overview

This project demonstrates the complete workflow of a machine learning project, including Docker setup, GitHub workflows, and AWS integration.

## Steps

1. **Docker Build** - Checked
2. **GitHub Workflow** - Configured
3. **IAM User in AWS** - Created

## Docker Setup on EC2

### Optional Commands

```bash
sudo apt-get update -y
sudo apt-get upgrade
```

### Required Commands

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

## Configure EC2 as Self-Hosted Runner

Follow these steps to configure your EC2 instance as a self-hosted runner for GitHub Actions.

## Setup GitHub Secrets

Ensure you have the following secrets set up in your GitHub repository:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` = `us-east-1`
- `AWS_ECR_LOGIN_URI` = `566373416292.dkr.ecr.ap-south-1.amazonaws.com`
- `ECR_REPOSITORY_NAME` = `simple-app`