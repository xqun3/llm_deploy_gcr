#!/bin/bash
DOCKER_REPO=${DOCKER_REPO:-"lmsysorg/sglang"}
DOCKER_VERSION=${DOCKER_VERSION:-"latest"}
REPO_NAMESPACE=${REPO_NAMESPACE:-"sagemaker_endpoint/vllm"}

# Get the ACCOUNT and REGION defined in the current configuration (default to us-west-2 if none defined)

ACCOUNT=${ACCOUNT:-$(aws sts get-caller-identity --query Account --output text)}
REGION=${REGION:-$(aws configure get region)}

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --region ${REGION} --repository-names "${REPO_NAMESPACE}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
echo "create repository:" "${REPO_NAMESPACE}"
aws ecr create-repository --region ${REGION} --repository-name "${REPO_NAMESPACE}" > /dev/null
fi

# if nwcd mirror
if [[ "$DOCKER_REPO" = 048912060910.dkr.ecr.cn-northwest-1.amazonaws.com.cn/* ]]; then
    aws ecr get-login-password --region cn-northwest-1 | docker login --username AWS --password-stdin 048912060910.dkr.ecr.cn-northwest-1.amazonaws.com.cn
fi

# Log into Docker
if [[ "$REGION" = cn* ]]; then
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com.cn
    REPO_NAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com.cn/${REPO_NAMESPACE}:${DOCKER_VERSION}"
else
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com
    REPO_NAME="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAMESPACE}:${DOCKER_VERSION}"
fi

echo ${REPO_NAME}

# Build docker
rm -rf app/.ipynb_checkpoints
docker build --build-arg DOCKER_VERSION=${DOCKER_VERSION} --build-arg DOCKER_REPO=${DOCKER_REPO} -t ${REPO_NAMESPACE}:${DOCKER_VERSION} .

# Push it
docker tag ${REPO_NAMESPACE}:${DOCKER_VERSION} ${REPO_NAME}
docker push ${REPO_NAME}
