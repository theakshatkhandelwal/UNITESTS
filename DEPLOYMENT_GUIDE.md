# 🚀 Deployment Guide - UniTest AI Learning Platform

This guide covers deployment options for your Flask application with PostgreSQL database.

## 🎯 **Recommended Platforms**

### 1. **Railway** ⭐ (Best Choice)
**Why Railway?**
- Excellent Flask support
- Automatic PostgreSQL database
- Simple deployment process
- Good performance
- Reasonable pricing

**Deployment Steps:**
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub** repository
3. **Create new project** from GitHub repo
4. **Add PostgreSQL** database:
   - Go to your project dashboard
   - Click "New" → "Database" → "PostgreSQL"
5. **Set Environment Variables**:
   ```
   SECRET_KEY=your-secret-key-here
   GOOGLE_AI_API_KEY=your-google-ai-api-key
   ```
6. **Deploy**: Railway will automatically deploy from your main branch

**Cost**: $5/month for paid plans (includes database)

---

### 2. **Fly.io** ⭐ (Great Performance)
**Why Fly.io?**
- Global edge deployment
- Excellent performance
- Generous free tier
- Good for scaling

**Deployment Steps:**
1. **Install Fly CLI**:
   ```bash
   # Windows (PowerShell)
   iwr https://fly.io/install.ps1 -useb | iex
   ```
2. **Sign up** at [fly.io](https://fly.io)
3. **Login**:
   ```bash
   fly auth login
   ```
4. **Launch app**:
   ```bash
   fly launch
   ```
5. **Add PostgreSQL**:
   ```bash
   fly postgres create
   fly postgres attach <postgres-app-name>
   ```
6. **Set secrets**:
   ```bash
   fly secrets set SECRET_KEY=your-secret-key
   fly secrets set GOOGLE_AI_API_KEY=your-google-ai-key
   ```
7. **Deploy**:
   ```bash
   fly deploy
   ```

**Cost**: Generous free tier, then pay-as-you-go

---

### 3. **DigitalOcean App Platform** ⭐ (Most Reliable)
**Why DigitalOcean?**
- Very reliable platform
- Excellent documentation
- Managed databases
- Good support

**Deployment Steps:**
1. **Sign up** at [DigitalOcean](https://digitalocean.com)
2. **Go to App Platform**
3. **Create new app** from GitHub
4. **Configure app** using `.do/app.yaml`
5. **Add managed database** (PostgreSQL)
6. **Set environment variables** in dashboard
7. **Deploy**

**Cost**: $12/month minimum (includes database)

---

## 🗄️ **PostgreSQL Database Setup**

### Option 1: Platform-Managed Database (Recommended)
Most platforms offer managed PostgreSQL:
- **Railway**: Automatic PostgreSQL addon
- **Fly.io**: `fly postgres create`
- **DigitalOcean**: Managed database in dashboard

### Option 2: External Database Services
If you need external database:
- **Neon** (Free tier available)
- **Supabase** (Free tier available)
- **PlanetScale** (Free tier available)

**Connection String Format:**
```
postgresql://username:password@host:port/database_name
```

---

## 🔧 **Environment Variables**

Set these in your platform's environment variables section:

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask secret key | Yes | `your-secret-key-here` |
| `GOOGLE_AI_API_KEY` | Google AI API key | Yes | `AIzaSy...` |
| `DATABASE_URL` | PostgreSQL connection | Yes | `postgresql://...` |
| `FLASK_ENV` | Environment | No | `production` |
| `PORT` | Server port | No | `8080` |

---

## 📋 **Pre-Deployment Checklist**

- [ ] Update `requirements.txt` with all dependencies
- [ ] Test app locally with PostgreSQL
- [ ] Set up Google AI API key
- [ ] Choose deployment platform
- [ ] Set up database
- [ ] Configure environment variables
- [ ] Test deployment

---

## 🚀 **Quick Start Commands**

### Railway
```bash
# No CLI needed - just connect GitHub repo
```

### Fly.io
```bash
fly auth login
fly launch
fly postgres create
fly secrets set SECRET_KEY=your-key
fly deploy
```

### DigitalOcean
```bash
# Use web dashboard or doctl CLI
doctl apps create --spec .do/app.yaml
```

---

## 🔍 **Troubleshooting**

### Common Issues:

1. **Database Connection Error**
   - Check `DATABASE_URL` format
   - Ensure database is running
   - Verify connection string

2. **Google AI API Error**
   - Verify API key is correct
   - Check API quotas
   - Ensure key has proper permissions

3. **Build Failures**
   - Check `requirements.txt`
   - Verify Python version compatibility
   - Check build logs

4. **Static Files Not Loading**
   - Ensure static files are in correct directory
   - Check platform-specific static file serving

---

## 💡 **Performance Tips**

1. **Database Optimization**
   - Use connection pooling
   - Add database indexes
   - Optimize queries

2. **Caching**
   - Implement Redis for session storage
   - Cache AI responses
   - Use CDN for static files

3. **Monitoring**
   - Set up error tracking (Sentry)
   - Monitor database performance
   - Track API usage

---

## 🎯 **Recommendation**

**For your use case, I recommend Railway** because:
- ✅ Excellent Flask support
- ✅ Automatic PostgreSQL setup
- ✅ Simple deployment process
- ✅ Good performance
- ✅ Reasonable pricing ($5/month)
- ✅ No complex configuration needed

**Next Steps:**
1. Sign up at [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Add PostgreSQL database
4. Set environment variables
5. Deploy!

Your app will be live in minutes! 🚀

