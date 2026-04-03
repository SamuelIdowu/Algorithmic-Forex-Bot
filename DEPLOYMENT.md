# 🚀 Deployment Guide — Algo Bot

This guide covers the best **free** ways to deploy your AI Trading Bot 24/7.

## 🏆 Recommendation: Oracle Cloud (Always Free)
This is the "gold standard" for free hosting. You get a massive amount of resources for $0.

- **CPU**: 4 ARM Neoverse N1 cores.
- **RAM**: 24 GB (more than enough for scikit-learn and multi-agent loops).
- **Storage**: 200 GB.
- **Persistence**: 100% (Your bot stays online even after you close your laptop).

### Steps
1. Create an account at [Oracle Cloud](https://www.oracle.com/cloud/free/).
2. Create an "Ampere" instance with **Ubuntu 22.04**.
3. Install Docker on the instance:
   ```bash
   curl -fsSL https://get.docker.com | sh
   ```
4. Clone your code and run:
   ```bash
   docker-compose up -d
   ```

---

## ⚖️ Comparison Table

| Feature | Oracle Cloud (Free) | Self-Hosting |
| :--- | :--- | :--- |
| **RAM** | 24 GB | Whatever you have |
| **Uptime** | 24/7 (Reliable) | 24/7 (Power dependent) |
| **Complexity** | Medium (Linux/Docker) | Medium (Maintenance) |
| **Persistent DB** | ✅ Yes | ✅ Yes |
| **Trading Agent** | ✅ Supported | ✅ Supported |

---

## 🔍 Pros & Cons

### 1. Oracle Cloud Free Tier
*   **Pros**: 
    - **Performance**: The ARM instances are faster and have more RAM than most paid small VPS plans.
    - **Persistence**: Your database stays on the disk; no data loss on restart.
    - **Isolation**: Runs everything (Bot, Telegram, Dashboard) in one place.
*   **Cons**:
    - **Setup**: Requires credit card verification and some Linux/Docker knowledge.
    - **Availability**: High demand means "Out of Capacity" errors can happen in popular regions during signup.

### 2. Self-Hosting (Old Laptop/Desktop)
*   **Pros**:
    - **Privacy**: Your data never leaves your house.
    - **No Signup**: No credit cards or identity verification.
*   **Cons**:
    - **Electricity**: Costs a small amount to run 24/7.
    - **Reliability**: If your Wi-Fi drops or power goes out, the bot stops trading.

---

---

## 🚀 Deployment: Render (One-Click)
Render is a great alternative to Oracle Cloud if you want a managed service.

### 1. Requirements
- A Render account.
- A GitHub/GitLab repository with your code.

### 2. Steps
1. Push your code to a private repository.
2. In Render, click **New +** and select **Blueprint**.
3. Connect your repository.
4. Render will automatically detect `render.yaml`.
5. Populate the **Environment Variables** (API keys) in the Render Dashboard.
6. Click **Deploy**.

### 3. Note on Persistence
The `render.yaml` includes a **Disk** definition for `/app/data`. 
> [!IMPORTANT]
> Render Disks are only available on paid plans (starting at $7/mo). For a 100% free setup, use Oracle Cloud or Self-Hosting as described above.

---

## 🐳 Running with Docker (Recommended)
We have provided a `Dockerfile` and `docker-compose.yml` to make everything plug-and-play.

### 1. Configure Secrets
Ensure your `.env` is populated with your API keys (Alpaca, Telegram, etc.).

### 2. Launch Everything
```bash
docker-compose up --build -d
```
This starts 2 containers:
- `agent`: The AI Trading loop.
- `telegram`: The interactive bot.

### 3. Check Logs
```bash
docker-compose logs -f
```

---

## ⚠️ Important Considerations
- **SQLite Database**: The bot uses a local SQLite file. If you move providers, remember to copy `data/market_data.db`.
- **API Keys**: Never commit your `.env` file to public GutHub repositories.
- **Memory**: Scikit-learn models (especially training) can spike RAM usage. Oracle's 24GB is ideal; Google's 1GB Free Tier may crash during training.
