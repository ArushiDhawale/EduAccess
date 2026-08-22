import os
import json
import re

# Central cache for vocabulary configuration
_vocabulary_cache = None

def load_sign_vocabulary(config_path="config/sign_vocabulary.json"):
    """
    Loads the sign vocabulary configuration file.
    Uses caching to avoid repeated reads.
    """
    global _vocabulary_cache
    if _vocabulary_cache is not None:
        return _vocabulary_cache
        
    if not os.path.exists(config_path):
        # Return empty dictionary if file doesn't exist
        return {}
        
    try:
        with open(config_path, "r") as f:
            _vocabulary_cache = json.load(f)
        return _vocabulary_cache
    except Exception as e:
        print(f"Error loading vocabulary configuration: {e}")
        return {}

def clear_vocabulary_cache():
    """Clears the cached vocabulary configuration (useful for testing)."""
    global _vocabulary_cache
    _vocabulary_cache = None

def text_to_gloss(text, config_path="config/sign_vocabulary.json"):
    """
    Converts English text into an ISL gloss sequence.
    Handles phrase matching, word matching, aliases, and unsupported words.
    
    Args:
        text: input English string
        config_path: path to vocabulary JSON
    Returns:
        dict: {
            "glosses": list of matching gloss strings,
            "unsupported": list of unsupported words
        }
    """
    vocabulary = load_sign_vocabulary(config_path)
    if not vocabulary:
        return {"glosses": [], "unsupported": [text] if text.strip() else []}
        
    # 1. Normalize the input text (lowercase, remove punctuation, strip)
    normalized = text.lower()
    # Replace punctuation with spaces to avoid joining words
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    if not normalized:
        return {"glosses": [], "unsupported": []}
        
    # 2. Identify all multi-word phrases in the vocabulary (both gloss, display name, and aliases)
    # We map normalized phrase string to the matching GLOSS key.
    phrase_map = {}
    for gloss_key, sign_info in vocabulary.items():
        # Check display name
        display = sign_info.get("display_name", "").lower()
        if " " in display:
            phrase_map[display] = gloss_key
            
        # Check aliases
        for alias in sign_info.get("aliases", []):
            alias_lower = alias.lower()
            if " " in alias_lower:
                phrase_map[alias_lower] = gloss_key
                
        # Check gloss key itself (e.g. GOOD_MORNING -> "good morning")
        gloss_phrase = gloss_key.replace("_", " ").lower()
        if " " in gloss_phrase:
            phrase_map[gloss_phrase] = gloss_key
            
    # Sort phrases by word count (or character length) in descending order
    # to match longer phrases first (greedy matching)
    sorted_phrases = sorted(phrase_map.keys(), key=lambda x: len(x.split()), reverse=True)
    
    # 3. Match and replace phrases in the normalized text with placeholders
    working_text = normalized
    replaced_phrases = []
    
    for phrase in sorted_phrases:
        # Match phrase as a whole word sequence using word boundaries \b
        # to avoid partial matching (e.g. "good morning" inside "good mornings")
        pattern = r'\b' + re.escape(phrase) + r'\b'
        if re.search(pattern, working_text):
            gloss = phrase_map[phrase]
            placeholder = f"__gloss_{gloss.lower()}__"
            working_text = re.sub(pattern, placeholder, working_text)
            
    # 4. Tokenize the text into individual words/placeholders
    tokens = working_text.split()
    
    glosses = []
    unsupported = []
    
    # Create single-word mapping for lookups
    single_word_map = {}
    for gloss_key, sign_info in vocabulary.items():
        # Add gloss key itself (lowercase)
        single_word_map[gloss_key.lower()] = gloss_key
        # Add display name (lowercase, no spaces)
        single_word_map[sign_info.get("display_name", "").lower()] = gloss_key
        # Add aliases (lowercase)
        for alias in sign_info.get("aliases", []):
            single_word_map[alias.lower()] = gloss_key
            
    # 5. Process each token
    for token in tokens:
        if token.startswith("__gloss_") and token.endswith("__"):
            # Extract gloss from placeholder (e.g. "__gloss_thank_you__" -> "THANK_YOU")
            gloss_extracted = token.replace("__gloss_", "").replace("__", "").upper()
            glosses.append(gloss_extracted)
        else:
            # Check single word lookup
            if token in single_word_map:
                glosses.append(single_word_map[token])
            else:
                unsupported.append(token)
                
    return {
        "glosses": glosses,
        "unsupported": unsupported
    }

def get_sign_video(gloss, config_path="config/sign_vocabulary.json"):
    """
    Gets the configured video asset path for a sign gloss.
    Checks if the video file actually exists.
    
    Args:
        gloss: upper-case gloss string (e.g. "HELLO")
        config_path: path to vocabulary JSON
    Returns:
        str: video file path or None if not configured/file missing
    """
    vocabulary = load_sign_vocabulary(config_path)
    sign_info = vocabulary.get(gloss)
    if not sign_info:
        return None
        
    video_path = sign_info.get("video")
    if not video_path:
        return None
        
    # Check if the video file exists on disk
    if os.path.exists(video_path):
        return video_path
        
    return None
