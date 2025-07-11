# EnlistAI - Your AI Interview Assistant 

An intelligent AI-powered interview preparation assistant that helps job seekers prepare for interviews at top companies. EnlistAI combines company-specific information with curated interview questions to provide personalized interview preparation guidance.

##  Features

- **Company-Specific Interview Questions**: Access curated interview questions from top companies
- **Real-time Company Data**: Fetch live company information including ratings, reviews, and industry details
- **AI-Powered Responses**: Get intelligent, contextual answers using advanced RAG (Retrieval-Augmented Generation)
- **Interactive Web Interface**: User-friendly Streamlit-based interface
- **Vector Search**: Efficient similarity search using FAISS for relevant question retrieval
- **Multi-Company Support**: Covers major tech companies, consulting firms, and financial institutions

##  Architecture

The project uses a sophisticated RAG architecture:

1. **Data Processing**: Interview questions are cleaned and preprocessed
2. **Vector Embeddings**: Questions are converted to embeddings using HuggingFace's sentence-transformers
3. **Vector Storage**: FAISS index for efficient similarity search
4. **LLM Integration**: ChatGroq (Llama-3.1-8b-instant) for intelligent response generation
5. **API Integration**: Real-time company data fetching from external APIs

##  Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- Groq API Key
- LangChain API Key (optional, for tracing)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/EnlistAI.git
   cd EnlistAI
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   ```

5. **Load FAISS Index**
   you can download it from the following link:
   

##  Usage

1. **Start the Streamlit application**
   ```bash
   streamlit run EnlistAI.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

3. **Use the application**
   - Enter the company name you want to prepare for
   - Get personalized interview questions and preparation tips
   - View company-specific information and insights

##  Project Structure

```
EnlistAI/
├── EnlistAI.py                    # Main Streamlit application
├── embedder.py                    # Vector embedding generation
├── combined_api.py                # API endpoints
├── company_api.py                 # Company-specific API functions
├── interviewques_api.py           # Interview questions API
├── data_preprocessing.ipynb       # Data cleaning and preprocessing
├── extract.ipynb                  # Data extraction notebook
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
├── cleaned_interview_questions.csv # Processed interview questions
├── cleaned_top_companies.csv     # Company information
└── faiss_index/                  # Vector index storage
    └── company_questions/
        ├── index.faiss
        └── index.pkl
```

##  Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLM access
- `LANGCHAIN_API_KEY`: LangChain API key for tracing (optional)

### Model Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `llama-3.1-8b-instant` via ChatGroq
- **Vector Store**: FAISS with cosine similarity
- **Retrieval**: Top-k=3 similar questions

##  Data Sources

- **Interview Questions**: Curated from multiple sources and companies
- **Company Information**: Real-time data from company APIs
- **Supported Companies**: 
  - Tech: Google, Microsoft, Amazon, Apple, Meta, Netflix, etc.
  - Consulting: Accenture, Deloitte, McKinsey, etc.
  - Finance: JPMorgan, Goldman Sachs, Morgan Stanley, etc.
  - Indian IT: TCS, Infosys, Wipro, HCL, etc.

## API Endpoints

The project includes several API endpoints:
## **api repo : https://github.com/ADITYA15062005/EnlistAI**

- `/company/{company_name}`: Fetch company-specific information
- `/questions/{company_name}`: Get interview questions for a company\

##  How It Works

1. **User Input**: User enters a company name
2. **Company Detection**: System identifies the company from input
3. **Data Retrieval**: Fetches company information from APIs
4. **Question Retrieval**: Uses vector similarity to find relevant questions
5. **AI Response**: Generates comprehensive interview preparation guide
6. **Display**: Shows formatted results with company info and questions


Made with ❤️ by [Team Hackstreet Boys]
