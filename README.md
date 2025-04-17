# Fashion Voice Assistant

A sophisticated fashion recommendation system with conversational AI. This project comes in two versions:

## 1. Flask Version (Original)

The original version uses:
- Flask backend
- FAISS vector similarity search
- Sentence Transformers for embeddings 
- OpenAI API for conversational responses
- Azure Speech Services for voice capabilities

### Features
- Semantic search with vector embeddings
- Preference-based filtering
- Match percentage calculation based on user preferences
- Conversational interface with LLM-powered responses
- Voice interaction support

### Requirements
- Python 3.6+
- Flask
- FAISS 
- Sentence Transformers
- OpenAI API key
- Python dependencies in requirements.txt

## 2. GitHub Pages Version (Static)

A simplified version built for static hosting on GitHub Pages:
- Completely client-side JavaScript implementation
- OpenAI API calls directly from the browser
- Simplified search without vector embeddings
- User provides their own OpenAI API key

### Features
- Works on static hosting (GitHub Pages)
- No server required
- Same user interface
- Similar but simplified functionality

### Setup
To use the GitHub Pages version:
1. Visit: https://[username].github.io/fashion-voice-assistant
2. Enter your OpenAI API key
3. Start chatting with the fashion assistant

The GitHub Pages version is in the `/docs` folder of this repository.

## Repository Structure

- `/` - Main directory with Flask application (original version)
- `/docs` - Static GitHub Pages version
- `/templates` - HTML templates for Flask application

## License

This project is open source and available under the MIT License. 