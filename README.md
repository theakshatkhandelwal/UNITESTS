# 🎓 UniTest - AI-Powered Learning Platform

<div align="center">

![UniTest Logo](https://img.shields.io/badge/UniTest-AI%20Learning%20Platform-blue?style=for-the-badge&logo=graduation-cap)

**An intelligent quiz platform that adapts to your learning level using AI**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Google AI](https://img.shields.io/badge/Google%20AI-Gemini-orange?style=flat-square&logo=google)](https://ai.google.dev)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3+-purple?style=flat-square&logo=bootstrap)](https://getbootstrap.com)

[🚀 Live Demo](https://unittest-ai.vercel.app) | [📖 Documentation](#features) | [🛠️ Installation](#installation) | [🚀 Deploy to Vercel](#deployment) | [🤝 Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 How It Works](#-how-it-works)
- [🛠️ Installation](#️-installation)
- [⚙️ Configuration](#️-configuration)
- [🚀 Usage](#-usage)
- [🎨 Screenshots](#-screenshots)
- [🏗️ Architecture](#️-architecture)
- [📚 API Documentation](#-api-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

### 🧠 **AI-Powered Question Generation**
- **Smart Content Creation**: Uses Google Gemini AI to generate contextually relevant questions
- **Adaptive Difficulty**: Questions automatically adjust based on your learning progress
- **Multiple Question Types**: Support for both Multiple Choice Questions (MCQ) and Subjective questions
- **Question Variety**: Each quiz generates unique questions to avoid repetition

### 🎯 **Intelligent Learning System**
- **Bloom's Taxonomy Integration**: Questions progress through cognitive levels (Remembering → Creating)
- **Difficulty Levels**: Choose from Beginner, Intermediate, or Difficult levels
- **Progress Tracking**: Monitor your advancement across different topics
- **Continue Learning**: Resume from where you left off with one click

### 📊 **Comprehensive Assessment**
- **AI-Powered Evaluation**: Subjective answers are evaluated using advanced AI
- **Detailed Feedback**: Get comprehensive explanations for correct and incorrect answers
- **Performance Analytics**: Track your scores and improvement over time
- **PDF Export**: Download quizzes for offline study

### 🎨 **Modern User Experience**
- **Dark/Light Mode**: Toggle between themes with a single click
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Intuitive Interface**: Clean, modern design with smooth animations
- **Accessibility**: Built with accessibility best practices

### 📚 **AI Learning Assistant**
- **Personalized Learning Paths**: Get customized study recommendations
- **Topic Exploration**: Learn any subject with AI-generated explanations
- **Interactive Content**: Engaging learning materials with visual aids
- **PDF Processing**: Upload PDFs to automatically extract topics

---

## 🎯 How It Works

### 1. **Topic Selection**
Choose any topic you want to learn about, or upload a PDF document for automatic topic extraction.

### 2. **Difficulty Selection**
Select your preferred difficulty level:
- **Beginner**: Basic facts and definitions
- **Intermediate**: Application and analysis
- **Difficult**: Critical thinking and synthesis

### 3. **AI Question Generation**
Our AI creates personalized questions based on:
- Your selected topic
- Chosen difficulty level
- Bloom's taxonomy principles
- Question type preferences

### 4. **Interactive Quiz Taking**
Answer questions with real-time feedback and progress tracking.

### 5. **Intelligent Assessment**
Get detailed results with:
- Score breakdown
- Correct answer explanations
- AI-evaluated subjective responses
- Learning recommendations

---

## 🛠️ Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package installer)
- Google AI API key

### Step 1: Clone the Repository
```bash
git clone https://github.com/theakshatkhandelwal/unittest-ai-quiz.git
cd unittest-ai-quiz
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
GOOGLE_AI_API_KEY=your-google-ai-api-key
DATABASE_URL=sqlite:///unittest.db
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask secret key for sessions | Yes | - |
| `GOOGLE_AI_API_KEY` | Google AI API key for question generation | Yes | - |
| `DATABASE_URL` | Database connection string | No | `sqlite:///unittest.db` |
| `PORT` | Port number for the application | No | `5000` |

### Google AI Setup
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Create a new project
3. Generate an API key
4. Add the key to your `.env` file

---

## 🚀 Usage

### Getting Started

1. **Sign Up**: Create a new account or log in
2. **Dashboard**: Access your personalized learning dashboard
3. **Start Quiz**: Choose a topic and difficulty level
4. **Take Quiz**: Answer questions and get instant feedback
5. **Track Progress**: Monitor your learning journey

### Key Features Usage

#### 🎯 **Creating a Quiz**
1. Click "Start New Quiz" on the dashboard
2. Enter your topic or upload a PDF
3. Select difficulty level (Beginner/Intermediate/Difficult)
4. Choose question type and count
5. Click "Generate AI Quiz"

#### 📊 **Viewing Progress**
- Access your learning progress from the dashboard
- See your current level for each topic
- Click "Continue Learning" to resume where you left off

#### 🌙 **Theme Switching**
- Click the sun/moon icon in the top navigation
- Your preference is automatically saved
- Switch between light and dark modes anytime

#### 🤖 **AI Learning Assistant**
1. Click "AI Learning" on the dashboard
2. Enter any topic you want to learn
3. Select your current level and learning style
4. Get personalized learning content

---

## 🎨 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x400/4285f4/ffffff?text=Dashboard+View)

### Quiz Interface
![Quiz](https://via.placeholder.com/800x400/34a853/ffffff?text=Quiz+Interface)

### Dark Mode
![Dark Mode](https://via.placeholder.com/800x400/1a1a1a/ffffff?text=Dark+Mode+Theme)

### AI Learning
![AI Learning](https://via.placeholder.com/800x400/ea4335/ffffff?text=AI+Learning+Assistant)

---

## 🏗️ Architecture

### Technology Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Database**: SQLite (development) / PostgreSQL (production)
- **AI**: Google Gemini AI
- **Authentication**: Flask-Login
- **PDF Processing**: PyPDF2
- **Text Processing**: NLTK

### Project Structure
```
unittest-ai-quiz/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── dashboard.html    # Dashboard page
│   ├── quiz.html         # Quiz creation page
│   ├── take_quiz.html    # Quiz taking page
│   └── ...
├── static/               # Static files (CSS, JS, images)
├── instance/             # Database files
└── README.md            # This file
```

### Database Schema
- **Users**: User accounts and authentication
- **Progress**: Learning progress tracking per topic
- **Sessions**: Temporary quiz data storage

---

## 📚 API Documentation

### Routes

#### Authentication
- `GET /` - Home page
- `GET /login` - Login page
- `POST /login` - User login
- `GET /signup` - Signup page
- `POST /signup` - User registration
- `GET /logout` - User logout

#### Quiz Management
- `GET /dashboard` - User dashboard
- `GET /quiz` - Quiz creation page
- `POST /quiz` - Generate new quiz
- `GET /take_quiz` - Take quiz page
- `POST /submit_quiz` - Submit quiz answers
- `GET /quiz_results` - View quiz results

#### Learning Features
- `POST /ai_learn` - AI learning content generation
- `POST /continue_learning` - Continue from saved progress
- `POST /next_level` - Advance to next difficulty
- `POST /retry_level` - Retry current level

#### Utilities
- `POST /upload_pdf` - PDF processing
- `GET /download_pdf` - Download quiz as PDF

---

## 🚀 Deployment

### Quick Deploy to Vercel + NeonDB

1. **Fork this repository** to your GitHub account
2. **Set up NeonDB database**:
   - Create account at [NeonDB](https://neon.tech/)
   - Create a new project
   - Copy the connection string
3. **Deploy to Vercel**:
   - Go to [Vercel](https://vercel.com/)
   - Import your GitHub repository
   - Set environment variables:
     - `SECRET_KEY`: Generate a random secret key
     - `GOOGLE_AI_API_KEY`: Your Google AI API key
     - `DATABASE_URL`: Your NeonDB connection string
   - Deploy!

### Pre-deployment Check
Run the deployment checker:
```bash
python deploy_vercel.py
```

### Detailed Deployment Guide
See [VERCEL_DEPLOYMENT_GUIDE.md](VERCEL_DEPLOYMENT_GUIDE.md) for comprehensive deployment instructions.

### SEO Optimization
The application includes:
- ✅ Meta tags and Open Graph tags
- ✅ Structured data (JSON-LD)
- ✅ XML sitemap
- ✅ Mobile-responsive design
- ✅ Google Search Console ready

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Write clear commit messages
- Add tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google AI** for providing the Gemini AI API
- **Bootstrap** for the responsive UI framework
- **Font Awesome** for the beautiful icons
- **Flask** community for the excellent web framework

---

## 📞 Support

If you have any questions or need help:

- 📧 **Email**: [your-email@example.com]
- 🐛 **Issues**: [GitHub Issues](https://github.com/theakshatkhandelwal/unittest-ai-quiz/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/theakshatkhandelwal/unittest-ai-quiz/discussions)

---

<div align="center">

**Made with ❤️ by [Akshat Khandelwal](https://github.com/theakshatkhandelwal)**

⭐ **Star this repository if you found it helpful!**

</div>