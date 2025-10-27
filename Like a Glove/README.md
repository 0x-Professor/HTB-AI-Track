# Like a Glove - HTB Challenge Writeup

## Challenge Description

The challenge gave us a file called `chal.txt` with 84 lines of word analogies. The description said:

> Words carry semantic information. Similar to how people can infer meaning based on a word's context, AI can derive representations for words based on their context too! However, the kinds of meaning that a model uses may not match ours. We've found a pair of AIs speaking in metaphors that we can't make any sense of! The embedding model is glove-twitter-25. Note that the flag should be fully ASCII and starts with 'htb{'.

## Understanding the Problem

The file had lines like:
```
Like non-mainstream is to efl, battery-powered is to?
Like sycophancy is to بالشهادة, cont is to?
Like беспощадно is to indépendance, rs is to?
```

It's basically word analogies - "A is to B as C is to ?". We need to find the missing word for each line, and all the answers together form the flag.

## My Initial Approach (Wrong Attempts)

### Attempt 1: ASCII Filtering Too Aggressively

At first, I thought since the flag needs to be "fully ASCII", I should force every answer to be ASCII. So I modified the code to search through the top 50-100 results and pick the FIRST ASCII word I found.

**Problem:** This gave me wrong answers! For example, some lines gave me things like "deus!nndepois", "twiitterox", "vawards" which were ASCII but completely wrong. The flag looked like gibberish:
```
htb{emedaradeus!nndepoisan-nisaouml****mrn_chwwhbq...
```

This didn't work because I was ignoring the BEST match (highest similarity score) and just grabbing whatever ASCII word appeared first in the results.

### Attempt 2: Removing Non-ASCII Characters

Then I tried just taking the top result and filtering OUT any non-ASCII characters. 

**Problem:** This lost important information! Many answers were full-width Japanese/Korean numbers like `１`, `０`, `４` which are part of the actual answer. By removing them, I lost parts of the flag.

### Attempt 3: Overthinking the Solution

I spent a lot of time trying to:
- Look at multiple top results
- Create complex filtering logic
- Search through top 100 results for "special" ASCII characters
- Try to pattern-match what kind of characters should appear

**Problem:** I was making it way too complicated! The solution was actually much simpler.

## The Correct Solution

After checking the official writeup, I realized the approach is straightforward:

1. **Load the GloVe Twitter-25 model** using gensim
2. **For each analogy**, use the formula: `result_vector = word3_vector + (word2_vector - word1_vector)`
3. **Find the most similar word** to that result vector (just take the TOP result, don't overthink it!)
4. **Convert full-width characters to ASCII** - this is the key! Many results have full-width numbers like `１` which need to be converted to `1`

### The Key Insight

The phrase "flag should be fully ASCII" doesn't mean we need to FIND ASCII words in the results. It means we need to CONVERT the results to ASCII! 

Full-width characters (like `１`, `０`, `５`) are Unicode characters commonly used in Japanese/Korean text. They look like regular numbers but have different Unicode codes. We just need to convert them:
- `０` (0xFF10) → `0` (0x30)
- `１` (0xFF11) → `1` (0x31)
- etc.

The conversion is simple: `chr(code - 0xFEE0)` for full-width ASCII variants.

## Final Solution Code

```python
import gensim.downloader as api
import re

def load_model(model_name='glove-twitter-25'):
    model = api.load(model_name)
    return model

def process_line(line, model):
    match = re.match(r"Like (.+?) is to (.+?), (.+?) is to\?", line.strip())
    if match:
        word1, word2, word3 = match.groups()
        vector1 = model[word1]
        vector2 = model[word2]
        vec_target = model[word3]
        
        # Calculate analogy: word3 + (word2 - word1)
        analogy_vector = vec_target + (vector2 - vector1)
        result = model.similar_by_vector(analogy_vector, topn=1)
        return result[0][0]

def process_file(filename, model):
    flag = ""
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                result = process_line(line, model)
                flag += result
    
    # Convert full-width to ASCII
    ascii_flag = ""
    for char in flag:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:  # Full-width ASCII range
            ascii_flag += chr(code - 0xFEE0)
        else:
            ascii_flag += char
    
    return ascii_flag
```

## The Flag

```
htb{...}
```

## What I Learned

1. **Don't overcomplicate things** - Sometimes the simple solution is the right one
2. **Read the requirements carefully** - "Fully ASCII" meant conversion, not filtering
3. **Understand Unicode** - Full-width characters are common in multilingual models
4. **Trust the model** - The top result is usually the best result, don't second-guess it
5. **Word embeddings are powerful** - GloVe can capture semantic relationships between words in vector space

## Resources

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Reference Writeup](https://a-z.fi/ctf-writeups/HTB-Like-A-Glove)
