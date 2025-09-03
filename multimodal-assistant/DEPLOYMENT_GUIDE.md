# 🚀 Deployment Guide

Hướng dẫn deploy Multimodal AI Assistant lên các platform khác nhau.

## 📋 **Pre-deployment Checklist**

### **1. Code Quality Check**
```bash
# Format và lint code
make fmt

# Run all tests
make test-cov

# Security scan
bandit -r app/
safety check
```

### **2. Build và Test Docker Image**
```bash
# Build production image
docker build -f docker/Dockerfile -t multimodal-assistant:latest .

# Test image locally
docker run -p 8000:8000 multimodal-assistant:latest

# Test health endpoint
curl http://localhost:8000/health
```

### **3. Environment Configuration**
```bash
# Create production .env file
cp .env.example .env.production

# Set production values
APP_ENV=production
APP_AUTH_TOKEN=your_secure_token_here
```

---

## 🐳 **Docker Deployment**

### **Local Production Setup**
```bash
# Production build
docker-compose -f docker/docker-compose.yml up -d

# Monitor logs
docker-compose -f docker/docker-compose.yml logs -f

# Health check
curl http://localhost:8000/health
```

### **Docker Hub Publishing**
```bash
# Login to Docker Hub
docker login

# Tag image
docker tag multimodal-assistant:latest yourusername/multimodal-assistant:v1.0.0
docker tag multimodal-assistant:latest yourusername/multimodal-assistant:latest

# Push to registry
docker push yourusername/multimodal-assistant:v1.0.0
docker push yourusername/multimodal-assistant:latest
```

---

## ☁️ **Cloud Deployment**

### **1. AWS EC2 Deployment**

#### **Setup EC2 Instance**
```bash
# Launch Ubuntu 22.04 LTS instance (t3.medium or larger)
# Configure security groups: 22 (SSH), 80, 443, 8000

# Connect and setup
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

#### **Deploy Application**
```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-assistant.git
cd multimodal-assistant

# Setup environment
cp .env.example .env
nano .env  # Configure production settings

# Deploy
docker-compose -f docker/docker-compose.yml up -d

# Setup reverse proxy (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/multimodal-assistant
```

#### **Nginx Configuration**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### **2. Google Cloud Platform (GCP)**

#### **Cloud Run Deployment**
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project your-project-id

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project-id/multimodal-assistant

# Deploy to Cloud Run
gcloud run deploy multimodal-assistant \
    --image gcr.io/your-project-id/multimodal-assistant \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 900 \
    --max-instances 10
```

#### **Set Environment Variables**
```bash
gcloud run services update multimodal-assistant \
    --set-env-vars APP_ENV=production,APP_AUTH_TOKEN=your_token \
    --region us-central1
```

### **3. Azure Container Instances**
```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name multimodal-rg --location eastus

# Deploy container
az container create \
    --resource-group multimodal-rg \
    --name multimodal-assistant \
    --image yourusername/multimodal-assistant:latest \
    --dns-name-label multimodal-assistant-unique \
    --ports 8000 \
    --memory 4 \
    --cpu 2 \
    --environment-variables APP_ENV=production
```

---

## 🔧 **Production Optimizations**

### **1. Performance Tuning**
```yaml
# docker-compose.yml production overrides
version: "3.9"
services:
  api:
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
        reservations:
          memory: 2G
          cpus: "1.0"
    environment:
      - WORKERS=2
      - MAX_MEMORY_USAGE=3584  # 3.5GB
```

### **2. Health Monitoring**
```bash
# Add health check script
echo '#!/bin/bash
curl -f http://localhost:8000/health || exit 1' > health_check.sh
chmod +x health_check.sh

# Setup cron job
echo "*/5 * * * * /path/to/health_check.sh" | crontab -
```

### **3. Log Management**
```yaml
# docker-compose.yml logging
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### **4. SSL/HTTPS Setup**
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 📊 **Monitoring & Maintenance**

### **System Monitoring**
```bash
# Monitor Docker containers
docker stats

# Monitor system resources
htop
df -h
free -h

# Monitor application logs
docker-compose logs -f --tail=100
```

### **Application Monitoring**
```bash
# API health checks
curl -f http://your-domain.com/health

# Monitor response times
time curl http://your-domain.com/health

# Check model loading
curl -X POST http://your-domain.com/v1/multimodal/analyze \
    -F 'text=test' -F 'tasks=["summary"]'
```

### **Backup & Recovery**
```bash
# Backup RAG storage
docker-compose exec api tar -czf /tmp/rag_backup.tar.gz /app/storage/rag
docker cp $(docker-compose ps -q api):/tmp/rag_backup.tar.gz ./rag_backup.tar.gz

# Backup database (if using)
docker-compose exec db pg_dump -U user database > backup.sql
```

---

## 🛡️ **Security Considerations**

### **1. Environment Security**
- Use strong, unique authentication tokens
- Keep dependencies updated
- Regular security scans
- Implement rate limiting
- Use HTTPS in production

### **2. Container Security**
```bash
# Scan for vulnerabilities
docker scan multimodal-assistant:latest

# Run as non-root user (already implemented)
# Minimize attack surface
# Keep base images updated
```

### **3. Network Security**
```bash
# Firewall configuration
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
sudo ufw deny 8000  # Only allow through nginx
```

---

## 🔄 **CI/CD Integration**

### **GitHub Actions Example**
See `.github/workflows/` directory for:
- Automated testing
- Docker build and push
- Security scanning
- Deployment to staging/production

### **Manual Deployment Steps**
```bash
# Production deployment checklist
1. git pull origin main
2. make test
3. docker-compose -f docker/docker-compose.yml pull
4. docker-compose -f docker/docker-compose.yml up -d
5. Health check
6. Monitor logs for issues
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Out of Memory**
```bash
# Increase Docker memory limits
# Monitor memory usage
docker stats --no-stream

# Optimize model loading
# Clear model cache if needed
rm -rf ~/.cache/huggingface/
```

#### **2. Model Loading Failures**
```bash
# Check disk space
df -h

# Check internet connectivity
curl -I https://huggingface.co

# Download models manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/trocr-base-printed')"
```

#### **3. Container Start Issues**
```bash
# Check logs
docker-compose logs api

# Debug container
docker run -it --rm multimodal-assistant:latest /bin/bash

# Check port conflicts
sudo netstat -tulpn | grep :8000
```

**📞 Support:** Nếu gặp issues, hãy check GitHub Issues hoặc tạo issue mới với logs chi tiết.
