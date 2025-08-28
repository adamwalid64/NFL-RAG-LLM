# NFL Fantasy Football AI Assistant - RAG-Powered Draft Intelligence System

An intelligent NFL fantasy football draft assistant that leverages Retrieval-Augmented Generation (RAG) to provide real-time, data-driven draft recommendations. This system combines web scraping, natural language processing, and advanced AI to analyze current NFL news, injuries, depth charts, and fantasy rankings, delivering personalized draft strategies that adapt to league settings and scoring formats.

## 🚀 Features

- **Real-time NFL Data Intelligence**: Web scraping system that continuously collects and processes NFL news, injury reports, and fantasy rankings
- **Advanced RAG-Powered Draft Assistant**: Retrieval-Augmented Generation system using OpenAI GPT-4 and FAISS that generates personalized draft strategies and risk assessments
- **Production-Ready AI Infrastructure**: Built-in cost optimization, multi-source validation, and scalable data pipeline for handling thousands of articles

## 🛠️ Tech Stack

- **Languages**: Python 3.13
- **AI/ML**: OpenAI GPT-4, LangChain, FAISS
- **Web Scraping**: Playwright, BeautifulSoup, Trafilatura
- **Data Processing**: Pandas, NumPy
- **Vector Operations**: OpenAI Embeddings API
- **Text Processing**: RecursiveCharacterTextSplitter
- **Data Storage**: Pickle, CSV

## 📋 Prerequisites

- Python 3.13+
- OpenAI API key
- Internet connection for web scraping
- Sufficient storage for data processing

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NFL-RAG-LLM
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 📁 Project Structure

```
NFL-RAG-LLM/
├── Data/                                    # Processed data files
│   ├── sentiment_articles_*.pkl            # Chunked documents
│   ├── sentiment_articles_*.csv            # Raw article data
│   └── sentiment_articles_*_original_docs.pkl  # Original documents
├── NFL-scrape.py                           # Web scraping module
├── load-sentiment.py                       # Data processing and chunking
├── NFL-Rag.py                              # Main RAG system
├── README.md                               # This file
└── venv/                                   # Virtual environment
```

## 🔧 Usage

### 1. Data Collection
Run the web scraping module to collect NFL articles:
```bash
python NFL-scrape.py
```

### 2. Data Processing
Process and chunk the collected data:
```bash
python load-sentiment.py
```

### 3. RAG System
Run the main RAG system for draft recommendations:
```bash
python NFL-Rag.py
```

## 🏗️ Architecture

### Data Pipeline
1. **Web Scraping** (`NFL-scrape.py`)
   - Uses Playwright for dynamic content
   - Trafilatura for content extraction
   - BeautifulSoup as fallback parser
   - Stores data in CSV format

2. **Data Processing** (`load-sentiment.py`)
   - Converts CSV to LangChain Documents
   - Implements intelligent text chunking
   - Creates metadata for each document
   - Serializes processed data

3. **RAG System** (`NFL-Rag.py`)
   - Loads processed documents
   - Creates vector embeddings using OpenAI
   - Builds FAISS vector store
   - Implements RetrievalQA chain
   - Generates draft recommendations

### Key Components
- **Document Chunking**: RecursiveCharacterTextSplitter with 500-character chunks and 100-character overlap
- **Vector Database**: FAISS for efficient similarity search
- **LLM Integration**: OpenAI GPT-4 with cost tracking
- **Retrieval System**: Top-10 similarity-based document retrieval

## 💰 Cost Management

The system includes built-in cost tracking:
- Token counting for input/output
- Cost estimation before API calls
- Embedding cost calculation
- Query cost monitoring

## 📊 Data Sources

The system scrapes data from various NFL and fantasy football sources to provide comprehensive coverage of:
- Player news and updates
- Injury reports and status
- Depth chart changes
- Fantasy rankings and ADP
- Team and coaching updates

## 🔍 Customization

### League Settings
The system can be customized for different league formats:
- Team size (8, 10, 12, 14 teams)
- Scoring formats (Standard, Half-PPR, PPR)
- Draft types (Snake, Auction)
- Roster configurations

### Model Configuration
- Adjustable chunk sizes and overlap
- Configurable retrieval parameters
- Model selection (GPT-4, GPT-3.5-turbo)
- Cost optimization settings

## 🚨 Error Handling

- Graceful fallback for failed web scraping
- Robust error handling for API failures
- Data validation and integrity checks
- Comprehensive logging throughout the pipeline

## 📈 Performance

- Efficient vector search using FAISS
- Optimized document chunking
- Parallel processing capabilities
- Memory-efficient data handling

## 🔒 Security

- API key management through environment variables
- No hardcoded credentials
- Secure data handling practices
- Privacy-conscious web scraping

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT models and embeddings
- LangChain for the RAG framework
- FAISS for efficient vector similarity search
- The open-source community for the various libraries used

## 📞 Support

For questions or issues:
1. Check the existing issues
2. Create a new issue with detailed information
3. Include error messages and system information

## 🔮 Future Enhancements

- Real-time data streaming
- Multi-sport support
- Advanced analytics dashboard
- Mobile application
- API endpoints for external integration
- Machine learning model fine-tuning

---

**Note**: This system requires an active OpenAI API key and may incur costs based on usage. Monitor your API usage and costs regularly.