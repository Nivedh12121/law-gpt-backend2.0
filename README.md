# 🚂 Law GPT Backend

AI-powered legal assistant backend for Indian law, ready for Railway deployment.

## 🌟 Features

- ⚖️ **Indian Legal Knowledge**: Constitutional law, IPC, CrPC coverage
- 🚀 **FastAPI Framework**: High-performance async API
- 📚 **Comprehensive Responses**: Detailed legal explanations
- 🔒 **CORS Enabled**: Ready for frontend integration
- 📖 **Auto Documentation**: Swagger UI at `/docs`

## 🚀 Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

### One-Click Deployment:
1. Click the Railway button above
2. Connect your GitHub account
3. Deploy automatically!

### Manual Deployment:
1. Fork this repository
2. Connect to Railway
3. Deploy with zero configuration

## 🔗 API Endpoints

- `GET /` - Health check
- `POST /chat` - Legal chat interface
- `GET /health` - Detailed status
- `GET /docs` - API documentation

## 💬 Example Usage

```bash
curl -X POST https://your-app.railway.app/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Section 302 IPC?"}'
```

## 📚 Legal Coverage

- **Constitutional Law**: Articles 14, 19, 21, 32
- **Indian Penal Code**: Sections 302, 420, etc.
- **Criminal Procedure**: Bail provisions, court procedures
- **Fundamental Rights**: Equality, liberty, justice

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Access at http://localhost:8001
```

## 🌐 Environment Variables

- `PORT` - Server port (automatically set by Railway)

## 📄 License

MIT License - Feel free to use for educational and legal assistance purposes.

## ⚠️ Disclaimer

This provides general legal information only. Always consult qualified legal professionals for specific advice.