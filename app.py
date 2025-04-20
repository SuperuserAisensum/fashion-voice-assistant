import json
import os
import re
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
import openai

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "fashion-chatbot-secret-key")

# Dictionary to store conversation state
# We'll use this instead of server-side sessions for simplicity
conversations = {}

# Initialize OpenAI client with SambaNova API endpoint
openai_client = openai.OpenAI(
    api_key="9b54b7a6-5505-409e-a8d8-8187ac8cad04",
    base_url="https://api.sambanova.ai/v1"
)

# Load fashion data
with open('fashion_data.json', 'r', encoding='utf-8') as f:
    fashion_data = json.load(f)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare data for embedding
product_texts = []
for product in fashion_data:
    # Create a rich text representation of each product
    text = f"Title: {product['title']}. Brand: {product['brand']}. "
    text += f"Description: {product['short_description']} {product['full_description']}. "
    
    # Add categories if available
    if 'categories' in product:
        text += f"Categories: {', '.join(product['categories'])}. "
    
    # Add tags if available
    if 'tags' in product:
        text += f"Tags: {', '.join(product['tags'])}. "
    
    # Add materials if available
    if 'materials' in product and isinstance(product['materials'], list):
        materials_text = ", ".join([f"{m['name']} ({m['percentage']}%)" for m in product['materials']])
        text += f"Materials: {materials_text}. "
    
    # Add care instructions if available
    if 'care_instructions' in product:
        text += f"Care: {', '.join(product['care_instructions'])}. "
    
    product_texts.append(text)

# Generate embeddings for all products
print("Generating embeddings for products...")
product_embeddings = model.encode(product_texts, show_progress_bar=True)

# Build FAISS index
dimension = product_embeddings.shape[1]  # Get the dimension of embeddings
faiss_index = faiss.IndexFlatL2(dimension)  # Renamed to avoid conflicts
faiss_index.add(np.array(product_embeddings).astype('float32'))

# Conversation state management
class ConversationState:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.user_preferences = {
            # Priority preferences
            "occasion": None,
            "budget": None,
            "clothing_type": None,
            "material": None,
            # Optional preferences
            "size": None,
            "color": None,
            "style": None
        }
        # Track if we have recommended products already
        self.products_recommended = False
        # Store previously shown products
        self.shown_products = set()
        # Track preference changes since last recommendation
        self.preferences_changed = False
        # Store previous state of preferences to detect changes
        self.previous_preferences = self.user_preferences.copy()
    
    def update_last_activity(self):
        self.last_updated = datetime.now()
    
    def add_message(self, role, content, products=None):
        """
        Add a message to the conversation history
        
        Args:
            role: 'user' or 'assistant'
            content: message text
            products: optional list of products included with the message
        """
        message = {"role": role, "content": content}
        
        # Store products with the message if provided
        if products:
            message["products"] = products
            # Track these products as shown
            for product in products:
                self.shown_products.add(product['title'])
            
            # Reset the preferences_changed flag when we show products
            if role == 'assistant':
                self.preferences_changed = False
                # Update previous preferences state
                self.previous_preferences = self.user_preferences.copy()
        
        self.messages.append(message)
        self.update_last_activity()
    
    def get_shown_product_titles(self):
        """Get list of product titles that have been shown to the user"""
        return list(self.shown_products)
    
    def update_preference(self, preference_type, value):
        """Update user preference and track if it's changed"""
        if preference_type in self.user_preferences:
            # Check if this is a new value
            current_value = self.user_preferences.get(preference_type)
            if current_value != value:
                # Update the preference
                self.user_preferences[preference_type] = value
                self.update_last_activity()
                print(f"Updated preference: {preference_type} = {value}")
                
                # Mark that preferences have changed
                self.preferences_changed = True
    
    def get_conversation_history(self):
        return self.messages[-10:]  # Return the last 10 messages for context
    
    def get_priority_preferences_count(self):
        """Count how many priority preferences the user has provided"""
        priority_prefs = ["occasion", "budget", "clothing_type", "material"]
        return sum(1 for k in priority_prefs if self.user_preferences.get(k))
    
    def has_enough_preferences(self):
        """Check if user has provided enough priority preferences to recommend products"""
        return self.get_priority_preferences_count() >= 4  # Changed from 3 to 4

def get_or_create_conversation(session_id):
    if session_id not in conversations:
        conversations[session_id] = ConversationState(session_id)
    return conversations[session_id]

def extract_price_threshold(query: str) -> Tuple[str, int, bool]:
    """
    Extract price threshold from the query
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (comparison_type, price_threshold, is_threshold_found)
    """
    # Patterns for different price thresholds
    above_patterns = [
        r'(di\s*atas|lebih\s*dari|>\s*|diatas|lebih\s*mahal\s*dari|minimal|min)\s*(\d[\d.,]*)\s*(rb|ribu|k|rb|ratus|juta|rupiah|idr|IDR)?',
        r'(harga|price).*?(>|lebih\s*dari|di\s*atas|diatas)\s*(\d[\d.,]*)\s*(rb|ribu|k|rb|ratus|juta|rupiah|idr|IDR)?'
    ]
    
    below_patterns = [
        r'(di\s*bawah|kurang\s*dari|<\s*|dibawah|lebih\s*murah\s*dari|maksimal|max)\s*(\d[\d.,]*)\s*(rb|ribu|k|rb|ratus|juta|rupiah|idr|IDR)?',
        r'(harga|price).*?(<|kurang\s*dari|di\s*bawah|dibawah)\s*(\d[\d.,]*)\s*(rb|ribu|k|rb|ratus|juta|rupiah|idr|IDR)?'
    ]
    
    # Check for "above" patterns
    for pattern in above_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 3:
                # Get the price value
                price_str = match.group(2).replace('.', '').replace(',', '')
                price = int(price_str)
                
                # Check for units
                unit = match.group(3).lower() if len(match.groups()) >= 3 and match.group(3) else ''
                if unit in ['rb', 'ribu', 'k']:
                    price *= 1000
                elif unit == 'juta':
                    price *= 1000000
                
                return 'above', price, True
            
    # Check for "below" patterns
    for pattern in below_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 3:
                # Get the price value from the correct group
                # In "below" patterns, the price is in group 2 or 3 depending on the pattern
                group_idx = 2
                price_str = match.group(group_idx).replace('.', '').replace(',', '')
                price = int(price_str)
                
                # Check for units
                unit_idx = group_idx + 1
                unit = match.group(unit_idx).lower() if len(match.groups()) >= unit_idx and match.group(unit_idx) else ''
                if unit in ['rb', 'ribu', 'k']:
                    price *= 1000
                elif unit == 'juta':
                    price *= 1000000
                
                return 'below', price, True
    
    # Simple budget extraction (e.g., "budget 300 ribu")
    budget_patterns = [
        r'(?:budget|harga|kisaran)?\s*(?:sekitar)?\s*(\d[\d.,]*)\s*(rb|ribu|k|ratus|juta|rupiah|idr|IDR)?',
        r'(?:budget|harga|kisaran)?\s*(?:sekitar)?\s*(rp\.?|idr)\s*(\d[\d.,]*)\s*(rb|ribu|k|ratus|juta)?'
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # For the first pattern
            if match.group(1):
                price_str = match.group(1).replace('.', '').replace(',', '')
                try:
                    price = int(price_str)
                    
                    # Check for units
                    unit = match.group(2).lower() if match.group(2) else ''
                    if unit in ['rb', 'ribu', 'k']:
                        price *= 1000
                    elif unit == 'juta':
                        price *= 1000000
                    
                    return 'below', price, True  # Assume users want products below their budget
                except ValueError:
                    pass
            
            # For the second pattern (with currency symbol)
            elif len(match.groups()) >= 2 and match.group(2):
                price_str = match.group(2).replace('.', '').replace(',', '')
                try:
                    price = int(price_str)
                    
                    # Check for units
                    unit = match.group(3).lower() if len(match.groups()) >= 3 and match.group(3) else ''
                    if unit in ['rb', 'ribu', 'k']:
                        price *= 1000
                    elif unit == 'juta':
                        price *= 1000000
                    
                    return 'below', price, True  # Assume users want products below their budget
                except ValueError:
                    pass
    
    return '', 0, False

def filter_products_by_price(products: List[Dict[str, Any]], comparison: str, threshold: int) -> List[Dict[str, Any]]:
    """
    Filter products based on price threshold
    
    Args:
        products: List of product dictionaries
        comparison: Type of comparison ('above' or 'below')
        threshold: Price threshold value
        
    Returns:
        Filtered list of products
    """
    filtered_products = []
    
    for product in products:
        if 'price' in product and 'amount' in product['price']:
            price = product['price']['amount']
            
            if comparison == 'above' and price >= threshold:
                filtered_products.append(product)
            elif comparison == 'below' and price <= threshold:
                filtered_products.append(product)
    
    return filtered_products if filtered_products else products

def extract_color_preference(query: str) -> str:
    """Extract color preference from user query"""
    color_patterns = [
        r'\b(merah|hitam|putih|biru|hijau|kuning|pink|purple|ungu|orange|jingga|coklat|abu-abu|abu|grey|gray|silver|gold|emas|perak|maroon|navy|tosca|toska|teal|fuchsia|magenta|mauve|cream|ivory|turquoise|aqua|beige|khaki|olive|tan|coral|burgundy|indigo|lavender|violet|brown|black|white|red|blue|green|yellow|pink|purple|orange)\b'
    ]
    
    for pattern in color_patterns:
        match = re.search(pattern, query.lower())
        if match:
            return match.group(1)
    
    return None

def extract_clothing_type(query: str) -> List[str]:
    """Extract clothing type preferences from user query"""
    clothing_patterns = {
        'top': r'\b(top|atasan|kemeja|blouse|shirt|kaos|tee|t-shirt|polo|sweater|cardigan|sweatshirt|hoodie|tank|crop)\b',
        'pants': r'\b(pants|celana|trouser|jeans|chino|legging|jogger|short|cullotes|culottes)\b',
        'dress': r'\b(dress|gaun|terusan|gown|mini dress|maxi dress|jumpsuit|romper|overalls)\b',
        'skirt': r'\b(skirt|rok|mini skirt|maxi skirt|flare skirt|pencil skirt)\b',
        'outerwear': r'\b(jacket|jaket|coat|blazer|parka|windbreaker|bomber|cardigan|vest|kimono|outer|outerwear)\b'
    }
    
    found_types = []
    for clothing_type, pattern in clothing_patterns.items():
        if re.search(pattern, query.lower()):
            found_types.append(clothing_type)
    
    return found_types

def extract_material_preference(query: str) -> List[str]:
    """Extract material preferences from user query"""
    material_patterns = [
        r'\b(cotton|katun|polyester|poly|wool|wol|fleece|flanel|flannel|denim|jeans|knit|rajut|satin|silk|sutra|linen|rayon|spandex|elastane|velvet|corduroy|canvas|terry|twill|scuba|jersey|leather|kulit|suede|chambray|chiffon|lycra|acrylic|akrilik|nylon|nilon)\b'
    ]
    
    found_materials = []
    for pattern in material_patterns:
        matches = re.finditer(pattern, query.lower())
        for match in matches:
            found_materials.append(match.group(1))
    
    return found_materials

def extract_occasion(query: str) -> str:
    """Extract occasion from user query"""
    occasion_patterns = {
        'formal': r'\b(formal|kantor|office|kerja|work|meeting|rapat|business|bisnis|professional|profesional)\b',
        'casual': r'\b(casual|kasual|santai|daily|sehari-hari|everyday|hangout|jalan-jalan|weekend|akhir pekan)\b',
        'party': r'\b(party|pesta|celebration|perayaan|event|acara|gala|dinner|makan malam|date|kencan)\b',
        'sport': r'\b(sport|olahraga|gym|workout|exercise|fitness|jogging|running|yoga|pilates|training)\b'
    }
    
    for occasion, pattern in occasion_patterns.items():
        if re.search(pattern, query.lower()):
            return occasion
    
    return None

def extract_size_preference(query: str) -> str:
    """Extract size preference from user query"""
    size_patterns = [
        r'\b(S|M|L|XL|XXL|XXXL|xs|small|medium|large|kecil|sedang|besar)\b'
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None

def get_preference_keywords(preferences):
    """Generate search keywords based on user preferences"""
    keywords = []
    
    if preferences['occasion']:
        keywords.append(preferences['occasion'])
    
    if preferences['clothing_type']:
        if isinstance(preferences['clothing_type'], list):
            keywords.extend(preferences['clothing_type'])
        else:
            keywords.append(preferences['clothing_type'])
    
    if preferences['color']:
        keywords.append(preferences['color'])
    
    return " ".join(keywords)

def retrieve_similar_products(query: str, k: int = 5, conversation_state=None) -> List[Dict[str, Any]]:
    """
    Retrieve k most similar products to the query with filtering based on user preferences
    
    Args:
        query: User query string
        k: Number of products to retrieve
        conversation_state: Current conversation state with user preferences
        
    Returns:
        List of product dictionaries
    """
    # Generate search query based on preferences if available
    if conversation_state and any(conversation_state.user_preferences.values()):
        # Add preference keywords to the query
        preference_keywords = get_preference_keywords(conversation_state.user_preferences)
        query = f"{query} {preference_keywords}"
    
    # Check if query contains price threshold
    comparison, threshold, has_threshold = extract_price_threshold(query)
    
    # If we have a price preference in conversation_state and not in query
    if not has_threshold and conversation_state and conversation_state.user_preferences['budget']:
        threshold = conversation_state.user_preferences['budget']
        has_threshold = True
        comparison = 'below'  # Assume users want products below their budget
    
    # Generate embedding for the query
    query_embedding = model.encode([query])
    
    # Retrieve more products initially if we need to filter
    retrieve_count = k * 5 if (has_threshold or conversation_state) else k
    
    # Search the FAISS index
    distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), retrieve_count)
    
    # Get the products
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(fashion_data):  # Ensure index is valid
            product = fashion_data[idx]
            # Add similarity score to the product
            product_with_score = product.copy()
            product_with_score['similarity_score'] = float(distances[0][i])
            results.append(product_with_score)
    
    # Apply filtering based on user preferences
    filtered_results = results
    
    # Apply price filtering if threshold was found
    if has_threshold and comparison:
        filtered_results = filter_products_by_price(filtered_results, comparison, threshold)
    
    # Apply occasion filtering if specified
    if conversation_state and conversation_state.user_preferences['occasion']:
        occasion = conversation_state.user_preferences['occasion']
        occasion_filtered = []
        for product in filtered_results:
            # Check for occasion in categories, tags, or description
            occasion_match = False
            
            # Check categories
            if 'categories' in product and any(occasion.lower() in cat.lower() for cat in product['categories']):
                occasion_match = True
            # Check tags
            elif 'tags' in product and any(occasion.lower() in tag.lower() for tag in product['tags']):
                occasion_match = True
            # Check description
            elif occasion.lower() in product['short_description'].lower() or occasion.lower() in product['full_description'].lower():
                occasion_match = True
            
            if occasion_match:
                occasion_filtered.append(product)
        
        # Only update if we found matches, otherwise keep original results
        if occasion_filtered:
            filtered_results = occasion_filtered
    
    # Apply clothing type filtering if specified
    if conversation_state and conversation_state.user_preferences['clothing_type']:
        clothing_types = conversation_state.user_preferences['clothing_type']
        if not isinstance(clothing_types, list):
            clothing_types = [clothing_types]
        
        type_filtered = []
        for product in filtered_results:
            # Check for type in categories or description
            type_match = False
            if 'categories' in product and any(any(ct.lower() in cat.lower() for ct in clothing_types) for cat in product['categories']):
                type_match = True
            elif any(ct.lower() in product['title'].lower() for ct in clothing_types):
                type_match = True
            
            if type_match:
                type_filtered.append(product)
        
        # Only update if we found matches, otherwise keep original results
        if type_filtered:
            filtered_results = type_filtered
    
    # If filtering removed all results, return the original ones with a warning
    if not filtered_results:
        print(f"Warning: No products found matching filters. Returning unfiltered results.")
        # Add a message about filtering to the first product if we have results
        if results:
            # Add a message about price filtering to the first product
            results[0]['filter_message'] = f"Tidak ada produk yang cocok dengan semua kriteria Anda. Menampilkan alternatif terdekat."
        return results[:k]
    
    # Calculate match percentage based on fulfilled criteria
    for product in filtered_results:
        match_score = 0
        criteria_count = 0
        
        # Add debug prints to track match calculation
        print(f"Calculating match for: {product['title']}")
        
        # Budget match (PRIORITY)
        if has_threshold:
            criteria_count += 1
            if comparison == 'below' and product['price']['amount'] <= threshold:
                match_score += 1
                print(f"  ✓ Budget match: {product['price']['amount']} <= {threshold}")
            elif comparison == 'above' and product['price']['amount'] >= threshold:
                match_score += 1
                print(f"  ✓ Budget match: {product['price']['amount']} >= {threshold}")
            else:
                print(f"  ✗ Budget mismatch: {product['price']['amount']} not {comparison} {threshold}")
        
        # Occasion match (PRIORITY)
        if conversation_state and conversation_state.user_preferences['occasion']:
            criteria_count += 1
            occasion = conversation_state.user_preferences['occasion']
            occasion_match = False
            
            # Check for occasion in categories
            if 'categories' in product and any(occasion.lower() in cat.lower() for cat in product['categories']):
                occasion_match = True
                print(f"  ✓ Occasion match in categories: {occasion}")
            # Check for occasion in tags
            elif 'tags' in product and any(occasion.lower() in tag.lower() for tag in product['tags']):
                occasion_match = True
                print(f"  ✓ Occasion match in tags: {occasion}")
            # Check for occasion in description
            elif occasion.lower() in product['short_description'].lower() or occasion.lower() in product['full_description'].lower():
                occasion_match = True
                print(f"  ✓ Occasion match in description: {occasion}")
            else:
                print(f"  ✗ Occasion mismatch: {occasion} not found")
            
            if occasion_match:
                match_score += 1
        
        # Clothing type match (PRIORITY)
        if conversation_state and conversation_state.user_preferences['clothing_type']:
            criteria_count += 1
            clothing_types = conversation_state.user_preferences['clothing_type']
            if not isinstance(clothing_types, list):
                clothing_types = [clothing_types]
            
            type_match = False
            # Check categories
            if 'categories' in product and any(any(ct.lower() in cat.lower() for ct in clothing_types) for cat in product['categories']):
                type_match = True
                matched_cats = [cat for cat in product.get('categories', []) 
                               if any(ct.lower() in cat.lower() for ct in clothing_types)]
                print(f"  ✓ Type match in categories: {', '.join(matched_cats)}")
            # Check title
            elif any(ct.lower() in product['title'].lower() for ct in clothing_types):
                type_match = True
                matched_types = [ct for ct in clothing_types if ct.lower() in product['title'].lower()]
                print(f"  ✓ Type match in title: {', '.join(matched_types)}")
            else:
                print(f"  ✗ Type mismatch: {', '.join(clothing_types)} not found")
            
            if type_match:
                match_score += 1
        
        # Material match (PRIORITY)
        if conversation_state and conversation_state.user_preferences['material']:
            criteria_count += 1
            materials = conversation_state.user_preferences['material']
            if not isinstance(materials, list):
                materials = [materials]
            
            material_match = False
            if 'materials' in product and any(any(m.lower() in mat['name'].lower() for m in materials) for mat in product['materials']):
                material_match = True
                matched_mats = [mat['name'] for mat in product.get('materials', []) 
                               if any(m.lower() in mat['name'].lower() for m in materials)]
                print(f"  ✓ Material match: {', '.join(matched_mats)}")
            else:
                print(f"  ✗ Material mismatch: {', '.join(materials)} not found")
            
            if material_match:
                match_score += 1
        
        # Size match (OPTIONAL)
        if conversation_state and conversation_state.user_preferences['size']:
            size = conversation_state.user_preferences['size']
            size_match = False
            
            if 'sizes' in product and any(size.lower() == s['size'].lower() for s in product['sizes']):
                size_match = True
                print(f"  ✓ Size match: {size}")
                # Add as bonus but don't increase criteria count (optional)
                match_score += 0.5
            else:
                print(f"  ✗ Size mismatch: {size} not found")
        
        # Calculate percentage match
        if criteria_count > 0:
            product['match_percentage'] = int((match_score / criteria_count) * 100)
            print(f"  Match result: {match_score}/{criteria_count} = {product['match_percentage']}%")
        else:
            # Don't set a default match percentage, just use similarity score for ordering
            print("  No criteria to match against, using similarity score only")
    
    # Sort by match percentage (if available) and then by similarity score
    filtered_results.sort(key=lambda x: (x.get('match_percentage', 0), -x.get('similarity_score', 0)), reverse=True)
    
    return filtered_results[:k]

def generate_response_with_llm(query: str, conversation_state, products: List[Dict[str, Any]] = None) -> str:
    """
    Generate a conversational response using OpenAI's LLM
    
    Args:
        query: User query string
        conversation_state: Current conversation state
        products: List of retrieved products (optional)
        
    Returns:
        Generated response
    """
    try:
        # Get conversation history for context
        messages = conversation_state.get_conversation_history()
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-5:]])  # Last 5 messages for brevity
        
        # Check preferences
        preferences = conversation_state.user_preferences
        
        # Count actual preferences the user has provided (focus on materials, categories, occasions, budget)
        priority_preferences = {k: v for k, v in preferences.items() 
                               if k in ['occasion', 'budget', 'clothing_type', 'material']}
        preference_count = sum(1 for v in priority_preferences.values() if v)
        has_enough_preferences = preference_count >= 4
        
        # Detect if user is expressing satisfaction or wants to end conversation
        is_satisfied = any(word in query.lower() for word in ['terima kasih', 'makasih', 'thanks', 'cukup', 'sudah cukup', 'ok', 'oke', 'baik', 'good', 'bagus', 'keren', 'sip'])
        
        # Check if user is asking for different/more recommendations
        is_asking_different = re.search(r'\b(rekomendasi lain|rekomendasi yang lain|pilihan lain|produk lain|yang lain|alternatif|berbeda|ganti|beda|model lain)\b', query.lower()) is not None
        is_asking_more = re.search(r'\b(ada lagi|rekomendasi lagi|masih ada|selain itu|rekomendasi baru|produk baru|more)\b', query.lower()) is not None
        
        # Set the request type
        if is_asking_different:
            request_type = "different"
        elif is_asking_more:
            request_type = "more"
        elif conversation_state.products_recommended:
            request_type = "followup"
        else:
            request_type = "initial"
        
        # Build detailed product context if we have products AND enough preferences
        product_context = ""
        if products and len(products) > 0 and (has_enough_preferences or conversation_state.products_recommended or is_asking_different or is_asking_more):
            # Select top 3 products for detailed info
            top_products = products[:3]
            
            # Provide detailed information about each product
            for idx, product in enumerate(top_products):
                product_context += f"\nProduct {idx+1}: {product['title']}\n"
                product_context += f"Brand: {product['brand']}\n"
                product_context += f"Price: {product['price']['currency']} {product['price']['amount']:,}\n"
                product_context += f"Description: {product['full_description']}\n"
                
                # Add material details
                if 'materials' in product and product['materials']:
                    materials = [f"{m['name']} ({m['percentage']}%)" for m in product['materials']]
                    product_context += f"Materials: {', '.join(materials)}\n"
                
                # Add size information
                if 'sizes' in product and product['sizes']:
                    sizes = [size['size'] for size in product['sizes']]
                    product_context += f"Available sizes: {', '.join(sizes)}\n"
                
                # Add categories
                if 'categories' in product and product['categories']:
                    product_context += f"Categories: {', '.join(product['categories'])}\n"
                
                # Add tags
                if 'tags' in product and product['tags']:
                    product_context += f"Tags: {', '.join(product['tags'])}\n"
                
                # Add style_with
                if 'style_with' in product and product['style_with']:
                    product_context += f"Style with: {', '.join(product['style_with'])}\n"
                
                # Add seasons
                if 'seasons' in product and product['seasons']:
                    product_context += f"Seasons: {', '.join(product['seasons'])}\n"
                
                # Add URL for viewing
                if 'url' in product:
                    product_context += f"Product link: {product['url']}\n"
                
                # Add match percentage if available
                if 'match_percentage' in product:
                    product_context += f"Match percentage: {product['match_percentage']}%\n"
                
                # Add reasons why this product matches preferences - focusing on priority preferences
                reasons = []
                for key, value in priority_preferences.items():
                    if value and key not in ['price_comparison']:
                        if key == 'budget':
                            if product.get('price') and product['price'].get('amount') and conversation_state.user_preferences.get('budget'):
                                budget_value = conversation_state.user_preferences['budget']
                                price_comparison = conversation_state.user_preferences.get('price_comparison', 'below')
                                
                                if price_comparison == 'below' and product['price']['amount'] <= budget_value:
                                    reasons.append(f"matches budget: below {budget_value}")
                                elif price_comparison == 'above' and product['price']['amount'] >= budget_value:
                                    reasons.append(f"matches budget: above {budget_value}")
                        elif key == 'clothing_type':
                            clothing_types = value if isinstance(value, list) else [value]
                            if 'categories' in product and any(any(ct.lower() in cat.lower() for ct in clothing_types) for cat in product['categories']):
                                reasons.append(f"matches clothing type: {value}")
                            elif any(ct.lower() in product['title'].lower() for ct in clothing_types):
                                reasons.append(f"matches clothing type: {value}")
                        elif key == 'material':
                            materials = value if isinstance(value, list) else [value]
                            if 'materials' in product and any(any(m.lower() in mat['name'].lower() for m in materials) for mat in product['materials']):
                                reasons.append(f"matches material: {value}")
                        elif key == 'occasion':
                            if 'categories' in product and any(value.lower() in cat.lower() for cat in product['categories']):
                                reasons.append(f"matches occasion: {value}")
                            elif 'tags' in product and any(value.lower() in tag.lower() for tag in product['tags']):
                                reasons.append(f"matches occasion: {value}")
                            elif value.lower() in product['short_description'].lower() or value.lower() in product['full_description'].lower():
                                reasons.append(f"matches occasion: {value}")
                
                if reasons:
                    product_context += f"This product {', '.join(reasons)}\n"
        
        # Identify missing preferences - focus on priority preferences
        missing_preferences = []
        for key in ['occasion', 'budget', 'clothing_type', 'material']:
            if not preferences.get(key):
                missing_preferences.append(key)
        
        # Build prompt with varying style based on conversation state
        if is_satisfied:
            # Closing conversation prompt
            prompt = f"""
You are April, a friendly and personable fashion stylist for "This is April" fashion brand. You're chatting in a conversational Bahasa Indonesia style.

Chat history:
{conversation_history}

The user's latest message indicates they are satisfied or want to end the conversation: "{query}"

User's preferences: {', '.join([f"{k}: {v}" for k, v in preferences.items() if v])}

Write a warm, friendly closing response that:
1. Acknowledges their satisfaction in a natural way
2. Thanks them for chatting in a conversational tone
3. Briefly mentions their style preferences (don't just list them all)
4. Encourages them to check out the recommended products with enthusiasm
5. Invites them to chat again when they need fashion advice

Be warm, genuine and natural - like a real fashion stylist wrapping up a productive styling session with a client they care about. Use conversational Bahasa Indonesia with natural expressions.
"""
        elif is_asking_different and products:
            # Different product recommendation prompt
            prompt = f"""
You are April, a knowledgeable fashion stylist for "This is April" fashion brand. You're having a friendly chat in Bahasa Indonesia. Your tone is casual, warm and conversational.

Chat history:
{conversation_history}

User's latest message: "{query}"
The user is asking for DIFFERENT recommendations from what you've already shown. They want to see alternative products.

Based on their preferences:
{', '.join([f"{k}: {v}" for k, v in preferences.items() if v])}

I've found some DIFFERENT products that might match their preferences:
{product_context}

Write a response that:
1. Acknowledges they want to see different options
2. Presents these alternative products enthusiastically
3. For each product, explain why it's a good match for their specific preferences 
4. Focus on benefits and how the products will look/feel for them
5. Ask for their feedback on these new options

Use conversational Bahasa Indonesia with natural expressions. Sound like a helpful friend giving fashion advice, not a catalog.

DON'T use phrases like "Berikut adalah rekomendasi alternatif" or "Sesuai permintaan Anda". Instead use more natural language like "Aku punya beberapa pilihan berbeda nih yang kayaknya cocok buat kamu" or "Kalau kamu mau coba style yang lain, coba lihat ini deh".
"""
        elif is_asking_more and products:
            # More product recommendation prompt
            prompt = f"""
You are April, a knowledgeable fashion stylist for "This is April" fashion brand. You're having a friendly chat in Bahasa Indonesia. Your tone is casual, warm and conversational.

Chat history:
{conversation_history}

User's latest message: "{query}"
The user is asking for MORE recommendations similar to what you've already shown. They want to see additional options.

Based on their preferences:
{', '.join([f"{k}: {v}" for k, v in preferences.items() if v])}

I've found some MORE products that might match their preferences:
{product_context}

Write a response that:
1. Acknowledges their request for more options in a natural way
2. Presents these additional products with enthusiasm
3. For each product, mention a key feature or benefit
4. Focus on how these new options complement what you've already shown
5. Ask for their feedback on these additional options

Use conversational Bahasa Indonesia with natural expressions. Sound excited to show them more options!

DON'T use phrases like "Berikut adalah rekomendasi tambahan" or "Sesuai permintaan Anda". Instead use more natural language like "Aku masih punya beberapa pilihan keren lainnya nih" or "Aku juga punya beberapa item lain yang kayaknya cocok banget buat kamu".
"""
        elif request_type == "followup" and products:
            # Follow-up product recommendation after continuing conversation
            prompt = f"""
You are April, a knowledgeable fashion stylist for "This is April" fashion brand. You're having a friendly chat in Bahasa Indonesia. Your tone is casual, warm and conversational.

Chat history:
{conversation_history}

User's latest message: "{query}"
The user has continued the conversation after you already showed them some products before.

Based on their preferences:
{', '.join([f"{k}: {v}" for k, v in preferences.items() if v])}

I've found some more products that might match their preferences:
{product_context}

Write a response that:
1. Responds naturally to what the user just said
2. Subtly transitions to introducing these new product recommendations 
3. Present the products in a conversational way
4. For each product, briefly explain why it's a good match
5. Ask for their feedback on these options in a natural, conversational way

Use conversational Bahasa Indonesia with natural expressions. Make the introduction of new products feel natural in the flow of conversation, not abrupt.

DON'T say things like "Berdasarkan preferensi Anda" or "Sesuai dengan preferensi yang Anda berikan" - instead, be more natural.
"""
        elif has_enough_preferences or (products and preference_count >= 2):
            # Initial product recommendation prompt
            prompt = f"""
You are April, a knowledgeable fashion stylist for "This is April" fashion brand. You're having a friendly chat in Bahasa Indonesia. Your tone is casual, warm and conversational.

Chat history:
{conversation_history}

User's latest message: "{query}"

Based on your conversation, you've learned these preferences:
{', '.join([f"{k}: {v}" for k, v in preferences.items() if v])}

I've found some products that might match their preferences:
{product_context}

Write a response that:
1. Responds naturally to what the user just said (don't ignore their message)
2. Presents 1-3 product recommendations in a conversational way
3. For each product, explain why it's a good match for their specific preferences
4. Instead of listing features, focus on benefits and how the product will look/feel for the user
5. Ask for their feedback on these options in a natural, conversational way
6. Use conversational Bahasa Indonesia with the natural flow and expressions a fashion stylist would use

Your recommendations should feel like a friend giving fashion advice, not like reading a catalog. Focus on why these items would look great on them based on their preferences.

DON'T say things like "Berdasarkan preferensi Anda" or "Sesuai dengan preferensi yang Anda berikan" - instead, be more natural like "Aku punya beberapa pilihan yang cocok dengan gaya casual yang kamu suka" or "Kayaknya kamu bakal suka dengan top hitam ini karena..."
"""
        else:
            # Need more information prompt - we don't have enough preferences yet
            prompt = f"""
You are April, a friendly fashion stylist for "This is April" fashion brand. You're having a casual styling conversation in Bahasa Indonesia.

Chat history:
{conversation_history}

User's latest message: "{query}"

So far, you've learned these preferences ({preference_count} out of minimum 4 needed):
{', '.join([f"{k}: {v}" for k, v in priority_preferences.items() if v])}

Missing preferences: {', '.join(missing_preferences)}

You need more information before making the perfect recommendation. You need to collect at least 4 of these key preferences (occasion, budget, clothing_type, material) before recommending products.

Write a natural, friendly response that:
1. Responds directly to their message in a personalized way
2. Asks about ONE specific preference from {', '.join(missing_preferences[:2])} in a natural, conversational way
3. Feels like a casual chat with a friendly fashion stylist, not an interrogation
4. Uses conversational Bahasa Indonesia with natural expressions and flow

DO NOT:
- Use formulaic phrases like "Untuk memberikan rekomendasi terbaik, saya perlu tahu..."  
- List or number your questions
- Ask more than one preference question at a time
- Sound robotic or formal

DO:
- Sound casual and friendly like "Kamu lebih suka warna apa nih untuk [item]?" or "Budget kamu untuk [item] sekitar berapa?"
- Connect your question to what they just said
- Use casual language like a friend would use

IMPORTANT: DO NOT recommend products yet! You need at least 4 key preferences from the user first.
"""

        # Call OpenAI LLM
        try:
            response = openai_client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct",
                messages=[
                    {"role": "system", "content": "You are April, a helpful, warm and enthusiastic fashion stylist who works for 'This is April' fashion brand. You chat in a conversational Bahasa Indonesia style like a friend would. Your personality is friendly, enthusiastic and you make people feel comfortable with your natural conversation style. Use casual, warm language with natural expressions like 'nih', 'dong', 'sih', 'banget', etc. where appropriate to sound authentic. Avoid sounding like you're reading from a script or catalog. IMPORTANT: DO NOT use any emoji, special characters, or Unicode symbols in your responses as they will not be properly processed by the text-to-speech system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent responses
                top_p=0.1
            )
            
            # Return generated response
            generated_text = response.choices[0].message.content.strip()
            return generated_text
        except openai.OpenAIError as oe:
            print(f"Error generating response with SambaNova API: {oe}")
            # Fallback response for SambaNova API errors
            return "Maaf, ada kendala dengan sistem chat kami. Coba refresh halaman atau periksa koneksi internet kamu."
    
    except Exception as e:
        print(f"Error generating response with LLM: {e}")
        # Simple fallback response
        return "Maaf, saya tidak dapat memberikan respon saat ini. Silakan coba lagi nanti."

def process_conversation(session_id, user_message):
    """
    Process user message in the context of the conversation
    
    Args:
        session_id: Unique session ID
        user_message: User's message
        
    Returns:
        Response dictionary
    """
    # Get or create conversation state
    conversation = get_or_create_conversation(session_id)
    
    # Add user message to conversation
    conversation.add_message("user", user_message)
    
    # Extract preferences from user message
    preferences = extract_preferences(user_message)
    
    # Track if we had preference changes
    had_preference_changes = False
    
    # Update user preferences in conversation state
    for key, value in preferences.items():
        if value:
            # Only update if we have a real value
            current_value = conversation.user_preferences.get(key)
            if current_value != value:
                conversation.update_preference(key, value)
                had_preference_changes = True
    
    # Check various types of user intents
    is_asking_different = check_if_asking_different(user_message)
    is_asking_for_more = check_if_asking_more(user_message)
    is_asking_recommendations = check_if_asking_recommendations(user_message)
    
    # Set request type for recommendations - this helps determine what products to show
    request_type = determine_request_type(
        is_asking_different, 
        is_asking_for_more, 
        is_asking_recommendations,
        conversation.products_recommended
    )
    
    # Create search query from preferences
    search_query = build_search_query(
        conversation.user_preferences, 
        user_message, 
        is_asking_different or is_asking_for_more
    )
    
    # Check if we should show products
    show_products = should_show_products(
        is_asking_different,
        is_asking_for_more,
        is_asking_recommendations,
        conversation
    )
    
    # Retrieve products based on preferences if appropriate
    products = []
    if search_query and show_products:
        products = retrieve_products_for_user(
            search_query, 
            conversation, 
            request_type,
            had_preference_changes
        )
    
    # Generate response using LLM
    response_text = generate_response_with_llm(user_message, conversation, products)
    
    # Add assistant response to conversation with products
    conversation.add_message("assistant", response_text, products=products if products else None)
    
    # Clean up old conversations (older than 1 hour)
    clean_old_conversations()
    
    # Prepare the response
    result = {
        "response": response_text,
        "conversation_state": {
            "session_id": conversation.session_id,
            "preferences": conversation.user_preferences
        }
    }
    
    # If we have products to show, include them in the response
    if products:
        # Log for debugging
        for idx, product in enumerate(products):
            print(f"Product {idx+1}: {product['title']} - Match Percentage: {product.get('match_percentage', 'N/A')}")
        
        # Include products in response
        result["products"] = products
        
        # Mark that we've recommended products
        conversation.products_recommended = True
    
    return result

def check_if_asking_different(message):
    """Check if user is asking for different products"""
    different_patterns = [
        r'\b(rekomendasi lain|rekomendasi yang lain|pilihan lain|produk lain|yang lain|alternatif|berbeda|ganti|beda)\b',
        r'\b(tidak suka|kurang suka|gak suka|ga suka|tidak cocok|kurang cocok|gak cocok|ga cocok)\b',
        r'\b(ada (yang|yg) lain|produk lainnya|model lain|model lainnya)\b'
    ]
    
    for pattern in different_patterns:
        if re.search(pattern, message.lower()):
            print("User is asking for different recommendations")
            return True
    return False

def check_if_asking_more(message):
    """Check if user is asking for more products"""
    more_patterns = [
        r'\b(ada lagi|rekomendasi lagi|masih ada|selain itu|rekomendasi baru|produk baru)\b',
        r'\b(mau (liat|lihat|liatin) (yang|yg) lain)\b',
        r'\b(show me more|more recommendations|more options)\b'
    ]
    
    for pattern in more_patterns:
        if re.search(pattern, message.lower()):
            print("User is asking for more recommendations")
            return True
    return False

def check_if_asking_recommendations(message):
    """Check if user is asking for recommendations"""
    recommendation_patterns = [
        r'\b(rekomendasi|recommend|sarankan|saran|suggest|produk|tunjukkan|tampilkan)\b',
        r'\b(apa (yang|yg) cocok|apa (yang|yg) bagus|apa (yang|yg) keren)\b',
        r'\b(tolong (kasih|beri))\b'
    ]
    
    for pattern in recommendation_patterns:
        if re.search(pattern, message.lower()):
            print("User is asking for recommendations")
            return True
    return False

def determine_request_type(is_asking_different, is_asking_more, is_asking_recommendations, has_recommended_before):
    """Determine the type of product request"""
    if is_asking_different:
        return "different"
    elif is_asking_more:
        return "more"
    elif is_asking_recommendations and has_recommended_before:
        print("User is asking for recommendations again, treating as 'more'")
        return "more"
    else:
        return "initial"

def build_search_query(preferences, user_message, is_asking_more_or_different):
    """Build search query from preferences and user message"""
    search_query = ""
    if any(preferences.values()):
        # Prioritize key preferences for search
        priority_keys = ['occasion', 'clothing_type', 'material', 'budget']
        
        search_terms = []
        # First add priority preferences
        for key in priority_keys:
            if preferences.get(key):
                val = preferences[key]
                if isinstance(val, list):
                    search_terms.extend([str(v) for v in val])
                else:
                    search_terms.append(str(val))
        
        # Then add any other preferences
        for key, val in preferences.items():
            if val and key not in priority_keys:
                if isinstance(val, list):
                    search_terms.extend([str(v) for v in val])
                else:
                    search_terms.append(str(val))
        
        search_query = " ".join(search_terms)
    
    # Add the user message for more specific search when asking for more/different
    if is_asking_more_or_different:
        search_query = f"{search_query} {user_message} {user_message}"
    
    return search_query

def should_show_products(is_asking_different, is_asking_for_more, is_asking_recommendations, conversation):
    """Determine if we should show products in the response"""
    enough_preferences = conversation.has_enough_preferences()
    priority_count = conversation.get_priority_preferences_count()
    
    if is_asking_different or is_asking_for_more:
        # If user is explicitly asking for more/different products, always show them
        return True
    elif is_asking_recommendations and priority_count >= 3:  # Keep this at 3 for explicit requests
        # If user is asking for recommendations and has some preferences
        return True
    elif enough_preferences and not conversation.products_recommended:
        # If user has enough preferences (4+) and hasn't seen products yet
        return True
    elif enough_preferences and conversation.preferences_changed:
        # If user has enough preferences and preferences have changed, update recommendations
        print("Updating recommendations due to preference changes")
        return True
    else:
        return False

def retrieve_products_for_user(search_query, conversation, request_type, preferences_changed=False):
    """Retrieve and filter products based on the request type"""
    print(f"Retrieving products for query: {search_query}")
    print(f"Priority preferences count: {conversation.get_priority_preferences_count()}")
    print(f"Request type: {request_type}")
    print(f"Preferences changed: {preferences_changed}")
    
    # Previously shown products to exclude
    shown_product_titles = conversation.get_shown_product_titles()
    print(f"Previously shown products: {shown_product_titles}")
    
    # Retrieve more products to have variety
    base_k = 15
    
    # Get products
    all_products = retrieve_similar_products(search_query, k=base_k, conversation_state=conversation)
    
    if not all_products:
        print("No products found matching the query")
        return []
    
    # When preferences change, prioritize showing new recommendations 
    # that better match the updated preferences
    if preferences_changed:
        # Sort products by match percentage to show the best matches for updated preferences
        all_products.sort(key=lambda x: (x.get('match_percentage', 0), -x.get('similarity_score', 0)), reverse=True)
        return all_products[:3]
    
    # Filter out previously shown products for 'more' or 'different' requests
    if request_type in ["more", "different"] and shown_product_titles:
        # Get products not shown before
        new_products = [p for p in all_products if p["title"] not in shown_product_titles]
        
        if new_products:
            # Use different products if available
            print(f"Found {len(new_products)} new products not shown before")
            products = new_products[:3]
        else:
            # If all products have been shown, cycle through them with an offset
            print("All products have been shown before, recycling with offset")
            # Use modulo to cycle through available products
            offset = len(shown_product_titles) % len(all_products)
            end_idx = min(offset + 3, len(all_products))
            products = all_products[offset:end_idx]
            
            # If we need more products to reach 3, wrap around
            if len(products) < 3 and len(all_products) > 3:
                remaining = 3 - len(products)
                products.extend(all_products[:remaining])
    else:
        # For initial request, just take the top products
        products = all_products[:3]
    
    print(f"Showing products: {[p['title'] for p in products]}")
    return products

def clean_old_conversations():
    """Remove conversation states older than 1 hour"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, conversation in conversations.items():
        # If last updated more than 1 hour ago
        if (current_time - conversation.last_updated).total_seconds() > 3600:
            expired_sessions.append(session_id)
    
    # Remove expired sessions
    for session_id in expired_sessions:
        conversations.pop(session_id, None)

def extract_preferences(query: str) -> Dict[str, Any]:
    """Extract all preferences from a query
    
    Args:
        query: User query string
        
    Returns:
        Dictionary of preferences
    """
    preferences = {}
    
    # Extract price (PRIORITY)
    comparison, threshold, has_threshold = extract_price_threshold(query)
    if has_threshold:
        preferences['budget'] = threshold
        preferences['price_comparison'] = comparison
        print(f"Extracted budget preference: {comparison} {threshold}")
    
    # Extract clothing type (PRIORITY)
    clothing_types = extract_clothing_type(query)
    if clothing_types:
        preferences['clothing_type'] = clothing_types
        print(f"Extracted clothing type preference: {clothing_types}")
    
    # Extract material (PRIORITY)
    materials = extract_material_preference(query)
    if materials:
        preferences['material'] = materials
        print(f"Extracted material preference: {materials}")
    
    # Extract occasion (PRIORITY)
    occasion = extract_occasion(query)
    if occasion:
        preferences['occasion'] = occasion
        print(f"Extracted occasion preference: {occasion}")
    
    # Extract color (OPTIONAL)
    color = extract_color_preference(query)
    if color:
        preferences['color'] = color
        print(f"Extracted optional color preference: {color}")
    
    # Extract size (OPTIONAL)
    size = extract_size_preference(query)
    if size:
        preferences['size'] = size
        print(f"Extracted optional size preference: {size}")
    
    return preferences

@app.route('/api/speech-token', methods=['GET'])
def get_speech_token():
    """Get Azure Speech Service token for client-side speech recognition"""
    try:
        # Azure Speech Service credentials from environment variables
        speech_key = os.environ.get('SPEECH_KEY')
        speech_region = os.environ.get('SPEECH_REGION')
        
        if not speech_key or not speech_region:
            return jsonify({
                "error": "Speech credentials are not configured in environment variables"
            }), 500
            
        return jsonify({
            "speechKey": speech_key,
            "region": speech_region
        })
        
    except Exception as e:
        print(f"Error getting speech token: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')
    session_id = data.get('session_id', '')
    
    if not query:
        return jsonify({"response": "Mohon masukkan pertanyaan Anda tentang produk fashion."})
    
    # If no session ID provided, generate one
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Process the message in conversation context
    response_data = process_conversation(session_id, query)
    
    # Add session ID to response
    response_data['session_id'] = session_id
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Fashion Chatbot is ready. Starting Flask server...")
    # By default, use rule-based responses unless USE_SAMBANOVA_LLM is true
    print(f"Using SambaNova LLM: {os.environ.get('USE_SAMBANOVA_LLM', 'true')}")
    app.run(debug=True) 
