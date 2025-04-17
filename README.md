# Fashion RAG Chatbot with SambaNova Integration

A sophisticated web-based chatbot that uses a Retrieval-Augmented Generation (RAG) approach with a fashion product database as its knowledge base. The chatbot can answer questions about fashion products, such as their prices, sizes, materials, and care instructions. It integrates with SambaNova's Llama-4-Maverick LLM for enhanced response generation.

## Features

- Vector-based semantic search using Sentence Transformers
- Fast similarity search with FAISS
- Integration with SambaNova Cloud for enhanced natural language generation
- Web interface with chat functionality
- Detailed product information display with similar product suggestions
- Natural language understanding for fashion-related queries
- Support for queries in Indonesian and English
- Fallback to rule-based responses when API is unavailable

## Requirements

- Python 3.6+
- Flask
- Sentence Transformers
- FAISS (Facebook AI Similarity Search)
- NumPy
- python-dotenv
- OpenAI library (for SambaNova API integration)
- SambaNova API key

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file by copying `.env.example`:

```bash
cp .env.example .env
```

4. Update the `.env` file with your SambaNova API key:

```
SAMBANOVA_API_KEY=your_api_key_here
USE_SAMBANOVA_LLM=true  # Set to 'false' to use rule-based responses
```

## Usage

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Interact with the chatbot by typing questions about fashion products. For example:
   - "Apa ada celana putih?"
   - "Berapa harga Marlow 30 Pants?"
   - "Terbuat dari bahan apa celana ini?"
   - "Bagaimana cara merawat produk ini?"
   - "Ada ukuran apa saja untuk celana putih?"
   - "Rekomendasi pakaian untuk acara semi-formal"
   - "Warna apa saja yang tersedia untuk celana?"

## How It Works

The chatbot uses a Retrieval-Augmented Generation (RAG) approach:

1. **Retrieval:** The system uses Sentence Transformers to convert product descriptions into vector embeddings, creating a semantic representation of each product. When a user submits a query, it's also converted to a vector embedding, and FAISS (Facebook AI Similarity Search) quickly finds the most similar products based on vector similarity.

2. **Response Generation:** 
   - If `USE_SAMBANOVA_LLM=true`: The system uses the retrieved product information to formulate a prompt for SambaNova's Llama-4-Maverick LLM, which generates a natural, contextually appropriate response.
   - If `USE_SAMBANOVA_LLM=false` or if the API call fails: The system falls back to rule-based response generation based on query categorization.

The system processes and understands various types of queries related to:
- Product information
- Pricing
- Sizes
- Materials
- Care instructions
- Colors
- Style recommendations

## SambaNova Integration

The chatbot uses SambaNova's Llama-4-Maverick-17B model through their API to generate natural-sounding responses. This integration:

1. Takes the retrieved product information and formats it into a context-rich prompt
2. Sends this prompt to the SambaNova API
3. Uses the generated response to provide more natural, conversational answers

To use this feature:
- Make sure you have a valid SambaNova API key
- Set `USE_SAMBANOVA_LLM=true` in your `.env` file

## Advantages of the RAG Approach

- Semantic understanding: The chatbot can understand the meaning behind queries, not just match keywords
- Fuzzy matching: Similar concepts can be matched even if different words are used
- Fast retrieval: FAISS provides optimized vector similarity search
- Natural responses: SambaNova's LLM provides human-like responses based on retrieved information

## Customization

You can customize the chatbot by:

1. Adding more products to the `fashion_data.json` file
2. Extending the query pattern recognition in the `categorize_query` function in `app.py`
3. Modifying the UI in `templates/index.html`
4. Using a different Sentence Transformer model (update the model name in `app.py`)
5. Adjusting SambaNova LLM parameters (temperature, top_p) for different response styles

## License

This project is open source and available under the MIT License. 