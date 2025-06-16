import requests
import json

def test_grammar_checker():
    # Test cases
    test_cases = [
        {
            "text": "i went to the store and bought some milk. um, it was like really expensive you know. its price was high!",
            "description": "Test case with multiple issues (filler words, capitalization, apostrophes)"
        },
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "description": "Test case with no issues"
        },
        {
            "text": "i'm going to the store and i'll buy some milk. its going to be expensive!!",
            "description": "Test case with capitalization and punctuation issues"
        },
        {
            "text": "well, you know, like, i think that um, the weather is nice today.",
            "description": "Test case with multiple filler words"
        }
    ]

    # API endpoint
    url = "http://localhost:5000/speech2text"

    # Test each case
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['description']}")
        print(f"Input text: {test_case['text']}")
        
        try:
            response = requests.post(url, json={"text": test_case["text"]})
            response.raise_for_status()
            
            result = response.json()
            print("\nResults:")
            print(json.dumps(result, indent=2))
            
            # Print a summary of corrections
            print("\nCorrections Summary:")
            for pair in result["sentence_pairs"]:
                if pair["distance"] > 0:
                    print(f"\nOriginal: {pair['original']}")
                    print(f"Corrected: {pair['corrected']}")
                    print(f"Changes made: {pair['distance']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_grammar_checker() 