# Fashion Voice Assistant - GitHub Pages Version

This is a client-side only version of the Fashion Voice Assistant that runs entirely in the browser using GitHub Pages. This version uses client-side JavaScript for all the functionality that was previously handled by the Flask backend.

## How to Set Up

1. Fork this repository
2. Go to Settings > Pages
3. Set the source to "Deploy from a branch"
4. Select the branch (usually "main") and folder ("/docs")
5. Click "Save"
6. Wait for GitHub to build and deploy your site
7. Access your site at https://[your-username].github.io/fashion-voice-assistant/

## Features

- All processing happens in the browser
- Uses client-side JavaScript to handle product search
- Uses the OpenAI API directly from the browser
- Works with GitHub Pages static hosting

## Requirements

- You need to provide your own OpenAI API key
- Modern browser with JavaScript enabled

## How It Works

This version uses:
1. Client-side vector search (simple keyword matching)
2. Direct API calls to OpenAI from the browser
3. Static JSON data files for products
4. LocalStorage to save preferences and API keys

## Note

This is a simplified version of the original Flask application. Some advanced features like vector similarity search with FAISS have been replaced with simpler JavaScript search algorithms. 