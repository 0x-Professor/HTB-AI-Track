import gensim.downloader as api
import re

def load_model(model_name='glove-twitter-25'):
    print(f"Loading {model_name} model...")
    model = api.load(model_name)
    print("Model loaded successfully!")
    return model

def get_word_vector(model, word):
    try:
        vector = model[word]
        return vector
    except KeyError:
        return None

def process_line(line, model):
    match = re.match(r"Like (.+?) is to (.+?), (.+?) is to\?", line.strip())
    if match:
        word1, word2, word3 = match.groups()
        vector1 = get_word_vector(model, word1)
        vector2 = get_word_vector(model, word2)
        vec_target = get_word_vector(model, word3)

        if vector1 is not None and vector2 is not None and vec_target is not None:
            # Calculate the analogy vector: word3 + (word2 - word1)
            analogy_vector = vec_target + (vector2 - vector1)
            result = model.similar_by_vector(analogy_vector, topn=1)
            return result[0][0]
        else:
            missing_words = [word for word, vec in zip([word1, word2, word3], [vector1, vector2, vec_target]) if vec is None]
            print(f"Warning: Words not found in model: {', '.join(missing_words)}")
            return "?"
    else:
        print(f"Warning: Line format incorrect: {line.strip()}")
        return "?"

def process_file(filename, model):
    flag = ""
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                result = process_line(line, model)
                flag += result
    
    # Convert full-width characters to ASCII
    ascii_flag = ""
    for char in flag:
        code = ord(char)
        # Full-width ASCII variants (0xFF01-0xFF5E)
        if 0xFF01 <= code <= 0xFF5E:
            ascii_flag += chr(code - 0xFEE0)
        else:
            ascii_flag += char
    
    return ascii_flag

def main():
    model = load_model()
    filename = 'chal.txt'
    flag = process_file(filename, model)
    print("\n" + "="*80)
    print("FLAG:")
    print("="*80)
    print(flag)
    print("="*80)

if __name__ == "__main__":
    main()
