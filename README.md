# mlops-pytorch-pipeline


## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop
- kubectl + Minikube (or any Kubernetes cluster)

### 1. Local training (no Docker)

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements/train.txt
python src/train.py
```

Reads hyperparameters from `configs/training_config.yaml`, logs JSON metrics per epoch, saves checkpoint to `checkpoints/`.

### 2. Local serving (no Docker)

```bash
pip install -r requirements/serve.txt
python src/serve.py
```

Exposes `GET /health` and `POST /predict` on port 8080.

### 3. Docker

```bash
# Build
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Train (mount data + checkpoints)
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1

# Serve
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-serve:v1

# Test
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

### 4. Kubernetes (Minikube)

```bash
# Point Docker CLI at Minikube's daemon, then build/load images
eval $(minikube -p minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

# After training completes:
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods -n ml-training

# Test
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Notes

- Training was verified with a reduced epoch count (1 epoch) due to CPU-only local hardware constraints; the pipeline fully supports the default 10-epoch configuration given more compute time or GPU access.
- `imagePullPolicy: Never` is used in K8s manifests since images are loaded locally into Minikube rather than pulled from a registry.