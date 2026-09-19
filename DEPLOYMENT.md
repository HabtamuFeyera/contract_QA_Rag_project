# 🚀 Free Deployment Guide for LexiRAG (ContractAdvisor-AI)

This guide provides step-by-step instructions for deploying **LexiRAG** (FastAPI backend + React 18 frontend + ChromaDB vector store) **completely free of charge** ($0/month, no credit card required).

---

## 📑 Deployment Options at a Glance

| Strategy | Frontend | Backend | Cost | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Recommended)** | **Render** (Static Site) | **Render** (Web Service) | **$0 / mo** | 1-Click full-stack deployment via `render.yaml` blueprint |
| **Option 2 (Fastest UI)** | **Vercel** (Edge CDN) | **Render** (Web Service) | **$0 / mo** | Zero frontend cold starts + blazing fast global CDN |
| **Option 3 (Heavy AI / 16GB RAM)** | **Hugging Face Spaces** | **Hugging Face Spaces** | **$0 / mo** | 16 GB free RAM, unified container running Docker |

---

## 🌟 Option 1: 1-Click Deployment on Render (Recommended)

Render allows you to deploy both the **FastAPI backend** and the **React frontend** simultaneously using the blueprint defined in [`render.yaml`](./render.yaml).

### Step 1: Click the Deploy Button
Click the badge below or navigate to [Render Blueprint Deploy](https://render.com/deploy?repo=https://github.com/HabtamuFeyera/contract_QA_Rag_project):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/HabtamuFeyera/contract_QA_Rag_project)

*(Alternatively, sign in to [Render](https://render.com) -> Click **New +** -> **Blueprint** -> Connect `HabtamuFeyera/contract_QA_Rag_project`)*.

### Step 2: Configure Environment Variables
Render will read `render.yaml` and create two services:
1. `lexirag-backend` (Free Web Service)
2. `lexirag-frontend` (Free Static Site)

Under **Environment Variables**:
- `GEMINI_API_KEY`: Enter your free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- `DEFAULT_MODEL`: Set to `gemini-1.5-flash` (fast, capable, free tier with generous limits).

### Step 3: Deploy
- Click **Apply**.
- Render will install Python dependencies, start the FastAPI server, compile the React production bundle, and issue free SSL certificates (`https://...onrender.com`).
- Both URLs will be visible in your Render Dashboard!

---

## ⚡ Option 2: Vercel (Frontend) + Render (Backend) (Fastest UI)

For ultra-low latency frontend serving without any sleep periods, deploy the frontend on Vercel and the backend on Render.

### Part A: Deploy Backend on Render
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect `https://github.com/HabtamuFeyera/contract_QA_Rag_project`.
3. Configure settings:
   - **Name**: `lexirag-backend`
   - **Language**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Plan**: `Free`
4. Add Environment Variables:
   - `PYTHON_VERSION` = `3.11.9`
   - `GEMINI_API_KEY` = `your-free-gemini-api-key` (Get free from [aistudio.google.com](https://aistudio.google.com/app/apikey))
   - `DEFAULT_MODEL` = `gemini-1.5-flash`
5. Click **Create Web Service**. Note your backend URL (e.g., `https://lexirag-backend.onrender.com`).

### Part B: Deploy Frontend on Vercel
1. Go to [Vercel](https://vercel.com) and click **Add New...** -> **Project**.
2. Import `HabtamuFeyera/contract_QA_Rag_project`.
3. In project settings:
   - **Root Directory**: Click edit and choose `frontend`.
   - **Framework Preset**: `Create React App`
4. Add Environment Variable:
   - `REACT_APP_BACKEND_URL`: `https://lexirag-backend.onrender.com` (your Render backend URL from Part A).
5. Click **Deploy**. In under 60 seconds, your legal cockpit will be live with a custom `vercel.app` domain!

---

## 🤗 Option 3: Hugging Face Spaces (16GB RAM Free CPU)

Hugging Face Spaces provides 16GB RAM and 2 vCPUs completely free, which is ideal for ChromaDB and heavy document ingestion.

1. Create a free account at [Hugging Face](https://huggingface.co).
2. Go to **Spaces** -> **Create new Space**.
3. Name: `lexirag-legal-ai`
4. License: `MIT`
5. Space SDK: Select **Docker** -> **Blank**.
6. Clone or link your GitHub repository:
   - Run in your terminal:
     ```bash
     git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/lexirag-legal-ai
     git push hf main
     ```
7. In the Space **Settings** -> **Repository Secrets**:
   - Add Secret: `GEMINI_API_KEY` = `your-free-gemini-key`
8. The multi-stage [`Dockerfile`](./Dockerfile) will automatically compile the React frontend, package the FastAPI backend, and host both seamlessly on port `7860`.

---

## 🔍 Post-Deployment Health Check

Once your backend is live, verify connectivity:

```bash
# 1. Verify health status
curl https://<your-backend-url>/health

# Expected response:
# {"status":"healthy","version":"2.0.0","project":"LexiRAG - Autonomous Contract Legal Intelligence","chroma_ready":true,"api_key_configured":true}

# 2. Test semantic query
curl -X POST https://<your-backend-url>/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the escrow amount?", "top_k": 3}'
```

---

## 💡 Free Tier Tips & Optimization
1. **Cold Starts on Render Free Tier**: Free Render web services spin down after 15 minutes of inactivity. The first request after sleep may take ~30–45 seconds to spin back up.
2. **Persistence**: In free cloud instances, files stored in `data/contracts/` persist during runtime. For enterprise persistent multi-tenant contract storage, configure an S3 bucket or Supabase volume.
3. **CORS**: The backend has permissive CORS pre-configured (`*`) to allow seamless communication between your frontend and backend domains.
