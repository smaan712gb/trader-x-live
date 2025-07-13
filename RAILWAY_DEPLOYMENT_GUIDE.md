# 🚀 Railway Deployment Guide - Trader-X Live Intelligence

## Quick Deploy to Railway (Emergency Backup)

### **Step 1: Railway Setup (2 minutes)**

1. **Go to Railway Dashboard:** https://railway.app/dashboard
2. **Create New Service:**
   - Click "New Project" or use existing project
   - Select "Deploy from GitHub repo" or "Empty Service"
   - Name: `trader-x-live`

### **Step 2: Environment Variables (3 minutes)**

In Railway dashboard, go to **Variables** tab and add:

```bash
# Required API Keys
XAI_API_KEY=your_grok4_api_key_here
FMP_API_KEY=your_fmp_api_key_here
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here

# Railway Configuration
PORT=8501
PYTHONPATH=/app
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
```

### **Step 3: Deployment Method**

#### **Option A: GitHub Integration (Recommended)**
1. **Push code to GitHub repo**
2. **Connect repo to Railway**
3. **Auto-deploy on push**

#### **Option B: Railway CLI (Fastest)**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Deploy from current directory
railway up
```

#### **Option C: Direct Upload**
1. **Zip the entire Trader-X folder**
2. **Upload to Railway dashboard**
3. **Deploy manually**

### **Step 4: Configure Start Command**

In Railway **Settings** → **Deploy**:
```bash
streamlit run web_dashboard.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

### **Step 5: Deploy & Test (5 minutes)**

1. **Click Deploy** in Railway dashboard
2. **Wait for build** (2-3 minutes)
3. **Get Railway URL** (auto-generated)
4. **Test access** in browser
5. **Share URL** with team

---

## 🎯 **Expected Railway URL Format:**
```
https://trader-x-live-production.up.railway.app
```

## 📊 **Post-Deployment Checklist:**

### **Immediate Tests:**
- [ ] **Dashboard loads** without errors
- [ ] **Stock symbol input** works (try NVDA)
- [ ] **Grok-4 Live Search** returns results
- [ ] **Charts display** properly
- [ ] **Technical analysis** generates signals

### **Team Access:**
- [ ] **Share URL** with all team members
- [ ] **Test concurrent access** (2-5 users)
- [ ] **Verify Live Search costs** are tracking
- [ ] **Document login process** (if any)

### **Performance Monitoring:**
- [ ] **Railway metrics** show healthy CPU/Memory
- [ ] **Response times** under 3 seconds
- [ ] **No error logs** in Railway dashboard
- [ ] **Live Search** completing within 30 seconds

---

## 💰 **Cost Expectations:**

### **Railway Hosting:**
- **Starter Plan:** $5/month
- **Pro Plan:** $20/month (recommended for team)
- **Resource usage:** Low to moderate

### **Live Search Usage:**
- **Cost per source:** $0.025
- **Typical analysis:** 15-25 sources ($0.38-$0.63)
- **Daily team usage:** $2-10/day
- **Monthly estimate:** $50-300/month

### **Total Monthly Cost:**
- **Railway:** $20/month
- **Live Search:** $50-300/month
- **Per team member:** $14-64/month (split among 2-5 people)

---

## 🔧 **Troubleshooting:**

### **Common Railway Issues:**

#### **Build Failures:**
```bash
# Check requirements.txt compatibility
pip install -r requirements.txt

# Test locally first
streamlit run web_dashboard.py --server.port=8501 --server.address=0.0.0.0
```

#### **Connection Refused:**
- **Check start command** includes `--server.address=0.0.0.0`
- **Verify PORT** environment variable is set
- **Ensure** app binds to Railway's $PORT

#### **Environment Variables:**
- **Double-check** all API keys are correct
- **No trailing spaces** in variable values
- **Restart deployment** after variable changes

#### **Memory Issues:**
- **Upgrade Railway plan** if needed
- **Optimize** chart data loading
- **Add caching** for repeated requests

---

## 📞 **Emergency Contacts:**

### **Railway Support:**
- **Dashboard:** https://railway.app/dashboard
- **Docs:** https://docs.railway.app
- **Discord:** Railway community

### **API Support:**
- **xAI (Grok-4):** Check API status
- **FMP:** Verify API key limits
- **Supabase:** Database connectivity

---

## 🚀 **Success Metrics:**

### **Deployment Success:**
- ✅ **Railway shows "Active"** status
- ✅ **URL accessible** from browser
- ✅ **Dashboard loads** within 5 seconds
- ✅ **Live Search working** with real data

### **Team Adoption:**
- ✅ **All team members** can access
- ✅ **Concurrent usage** works smoothly
- ✅ **Live Search results** are valuable
- ✅ **Trading decisions** being made

### **Business Value:**
- ✅ **Faster trading decisions** with Grok-4 intelligence
- ✅ **Better entry/exit timing** with technical analysis
- ✅ **Risk management** with proper signals
- ✅ **Cost-effective** compared to Bloomberg terminals

---

## 📈 **Next Steps After Deployment:**

1. **Monitor usage** for first week
2. **Gather team feedback** on features
3. **Optimize costs** based on usage patterns
4. **Add custom domain** if needed
5. **Set up monitoring** and alerts
6. **Plan feature enhancements** (NQ futures, etc.)

**🎯 Target: Get team back online within 30 minutes!**
