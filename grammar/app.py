from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Common filler words to check for
FILLER_WORDS = [
    r'\b(um|uh|like|you know|sort of|kind of|basically|actually|literally|honestly)\b',
    r'\b(well|so|right|okay|anyway|anyways)\b'
]

# Punctuation patterns
PUNCTUATION_PATTERNS = {
    'missing_period': r'[^.!?]\s*$',  # Missing period at end of sentence
    'double_punctuation': r'[.!?]{2,}',  # Multiple punctuation marks
    'missing_comma': r'\b(and|but|or|so|yet|for|nor)\s+[A-Z]',  # Missing comma before conjunction
    'incorrect_apostrophe': r'\b(its|whos|yours|theirs|hers|his)\b'  # Missing apostrophe in possessives
}

# Capitalization patterns
CAPITALIZATION_PATTERNS = {
    'sentence_start': r'(?:^|[.!?]\s+)([a-z])',  # Lowercase at sentence start
    'proper_nouns': r'\b(i|i\'m|i\'ve|i\'ll|i\'d)\b'  # Lowercase 'I'
}

def fix_sentence(sentence, issues):
    """Fix the sentence based on the identified issues."""
    # Sort issues by position in reverse order to avoid position shifting
    issues.sort(key=lambda x: x['position'], reverse=True)
    
    corrected = sentence
    
    # First pass: handle filler words and clean up spacing
    for issue in issues:
        if issue['type'] == 'filler_word':
            # Get the word and its surrounding context
            word = issue['word']
            start = issue['position']
            end = start + len(word)
            
            # Check if the word is at the start of the sentence
            if start == 0:
                # Remove the word and any following comma and space
                corrected = corrected[end:].lstrip()
                if corrected.startswith(','):
                    corrected = corrected[1:].lstrip()
            else:
                # Check if there's a comma before the word
                has_comma_before = corrected[start-1:start] == ','
                # Check if there's a comma after the word
                has_comma_after = corrected[end:end+1] == ','
                
                # Remove the word and handle commas
                if has_comma_before and has_comma_after:
                    # Remove word and one comma
                    corrected = corrected[:start-1] + corrected[end+1:]
                elif has_comma_before:
                    # Remove word and comma before
                    corrected = corrected[:start-1] + corrected[end:]
                elif has_comma_after:
                    # Remove word and comma after
                    corrected = corrected[:start] + corrected[end+1:]
                else:
                    # Just remove the word
                    corrected = corrected[:start] + corrected[end:]
                
                # Clean up any double spaces
                corrected = ' '.join(corrected.split())
    
    # Second pass: handle punctuation and capitalization
    for issue in issues:
        if issue['type'] == 'missing_period':
            # Add period at the end if not already there
            corrected = corrected.rstrip()
            if not corrected.endswith(('.', '!', '?')):
                corrected += '.'
        elif issue['type'] == 'double_punctuation':
            # Replace multiple punctuation with single, preserving the last one
            last_punct = corrected.rstrip()[-1] if corrected.rstrip()[-1] in '.!?' else '.'
            corrected = re.sub(r'[.!?]+$', last_punct, corrected)
        elif issue['type'] == 'missing_comma':
            # Add comma before conjunction
            corrected = re.sub(r'\b(and|but|or|so|yet|for|nor)\s+([A-Z])', r'\1, \2', corrected)
        elif issue['type'] == 'incorrect_apostrophe':
            # Fix apostrophes in possessives and contractions
            word = issue['text']
            if word == 'its':
                # Only change to it's if it's a contraction, not a possessive
                if re.search(r'\bits\b(?=\s+(?:going|gonna|will|would|has|have|had))', corrected):
                    corrected = re.sub(r'\bits\b', "it's", corrected)
            elif word == 'whos':
                corrected = re.sub(r'\bwhos\b', "who's", corrected)
            elif word == 'yours':
                corrected = re.sub(r'\byours\b', "your's", corrected)
            elif word == 'theirs':
                corrected = re.sub(r'\btheirs\b', "their's", corrected)
            elif word == 'hers':
                corrected = re.sub(r'\bhers\b', "her's", corrected)
            elif word == 'his':
                corrected = re.sub(r'\bhis\b', "his's", corrected)
        elif issue['type'] == 'sentence_start':
            # Capitalize first letter
            corrected = corrected[0].upper() + corrected[1:]
        elif issue['type'] == 'proper_nouns':
            # Capitalize 'I' and its contractions
            corrected = re.sub(r'\bi\b', 'I', corrected)
            corrected = re.sub(r"\bi'([a-z])\b", lambda m: f"I'{m.group(1)}", corrected)
    
    # Final cleanup
    # Remove any double spaces
    corrected = ' '.join(corrected.split())
    # Ensure proper spacing around punctuation
    corrected = re.sub(r'\s+([.,!?])', r'\1', corrected)
    # Ensure space after punctuation if not at end
    corrected = re.sub(r'([.,!?])([^\s])', r'\1 \2', corrected)
    # Remove any leading/trailing spaces
    corrected = corrected.strip()
    
    return corrected

def calculate_distance(original, corrected):
    """Calculate the Levenshtein distance between original and corrected text."""
    if len(original) == 0:
        return len(corrected)
    if len(corrected) == 0:
        return len(original)
    
    # Create a matrix of size (len(original) + 1) x (len(corrected) + 1)
    matrix = [[0 for _ in range(len(corrected) + 1)] for _ in range(len(original) + 1)]
    
    # Initialize first row and column
    for i in range(len(original) + 1):
        matrix[i][0] = i
    for j in range(len(corrected) + 1):
        matrix[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(original) + 1):
        for j in range(1, len(corrected) + 1):
            if original[i-1] == corrected[j-1]:
                matrix[i][j] = matrix[i-1][j-1]
            else:
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,  # deletion
                    matrix[i][j-1] + 1,  # insertion
                    matrix[i-1][j-1] + 1  # substitution
                )
    
    return matrix[len(original)][len(corrected)]

def check_grammar(text):
    issues = []
    
    # Check for filler words
    for pattern in FILLER_WORDS:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            issues.append({
                'type': 'filler_word',
                'word': match.group(),
                'position': match.start()
            })
    
    # Check punctuation
    for issue_type, pattern in PUNCTUATION_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            issues.append({
                'type': issue_type,
                'text': match.group(),
                'position': match.start()
            })
    
    # Check capitalization
    for issue_type, pattern in CAPITALIZATION_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            issues.append({
                'type': issue_type,
                'text': match.group(),
                'position': match.start()
            })
    
    return issues

@app.route('/speech2text', methods=['POST'])
def transcribe():
    try:
        # Get text from request
        text = request.json.get('text', '')
        if not text:
            return jsonify({"error": "No text received"}), 400

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_pairs = []
        total_distance = 0
        corrections_count = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Check grammar issues
            issues = check_grammar(sentence)
            
            # Fix the sentence based on issues
            corrected = fix_sentence(sentence, issues)
            
            # Calculate distance between original and corrected
            distance = calculate_distance(sentence, corrected)
            
            total_distance += distance
            if distance > 0:
                corrections_count += 1

            sentence_pairs.append({
                "original": sentence,
                "corrected": corrected,
                "distance": distance
            })

        avg_distance = total_distance / len(sentence_pairs) if sentence_pairs else 0

        output = {
            "sentence_pairs": sentence_pairs,
            "stats": {
                "average_distance": avg_distance,
                "sentences_corrected": corrections_count,
                "total_sentences": len(sentence_pairs)
            }
        }

        return jsonify(output)
            
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        return jsonify({"error": f"Error processing text: {str(e)}"}), 400

@app.route('/_ah/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)