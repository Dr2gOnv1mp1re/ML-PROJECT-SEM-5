# 🚀 Cloud Deployment Guide

This guide explains how to deploy the **Conversational Image Recognition Chatbot** online for free.

---

## ⚠️ Why Vercel Fails for PyTorch & YOLOv8
**Vercel** is designed for frontend static sites (Next.js/React) and micro-serverless functions with a strict **250 MB maximum uncompressed limit**. 
Because deep learning frameworks (`torch`, `torchvision`, `ultralytics`, `scikit-learn`) require over 1.5 GB of dependencies, Vercel throws:
```
Error: Serverless function size exceeds limit: 250MB
```

---

## 🌟 Option 1: Deploy on Render.com (Recommended — 100% Free)

Render provides free persistent hosting with Docker support that can run PyTorch and FastAPI without size restrictions.

### Quick Step-by-Step:
1. Go to **[render.com](https://render.com/)** and sign in with your GitHub account.
2. Click **New +** in the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your repository:
   ```
   https://github.com/Dr2gOnv1mp1re/ML-PROJECT-SEM-5
   ```
4. Configure the service:
   - **Name:** `conversational-image-chatbot`
   - **Language / Environment:** Select **Docker** (Render will automatically detect our `Dockerfile`).
   - **Region:** Any region (e.g., Oregon or Frankfurt).
   - **Instance Type:** **Free** ($0/month).
5. Click **Create Web Service**.
6. Render will build the Docker container and provide you with a live HTTPS URL (e.g. `https://conversational-image-chatbot.onrender.com`).

---

## 🤗 Option 2: Deploy on Hugging Face Spaces (Free 16GB RAM)

Hugging Face Spaces is designed specifically for machine learning applications and gives you 16GB RAM for free.

### Quick Step-by-Step:
1. Go to **[huggingface.co/spaces](https://huggingface.co/spaces)** and sign in.
2. Click **Create new Space**.
3. Fill in:
   - **Space Name:** `conversational-image-chatbot`
   - **License:** Apache 2.0 or MIT
   - **Space SDK:** Choose **Docker** -> **Blank**.
4. Clone or push the repository code to your new Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/conversational-image-chatbot
   git push space main
   ```
5. Hugging Face will automatically build and launch the app with a public shareable URL.

---

## 🚂 Option 3: Deploy on Railway.app

1. Go to **[railway.app](https://railway.app/)** and log in with GitHub.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select `Dr2gOnv1mp1re/ML-PROJECT-SEM-5`.
4. Railway will automatically detect the `Dockerfile` and deploy the application.
