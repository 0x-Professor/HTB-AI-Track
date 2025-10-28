"""
Lost in Hyperspace - CTF Challenge Solution
============================================
Extracts the flag from high-dimensional token embeddings by projecting them
into 2D space using PCA and traversing via nearest-neighbor chain.

The challenge description references tesseracts and shadows - hinting at
dimensionality reduction from 512D to 2D/3D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(npz_file='token_embeddings.npz'):
    """Load token embeddings from NPZ file."""
    data = np.load(npz_file, allow_pickle=True)
    tokens = ''.join(data['tokens'])
    embeddings = data['embeddings']
    print(f"[+] Loaded {len(tokens)} tokens with {embeddings.shape[1]}D embeddings")
    print(f"[+] Tokens: {tokens}")
    return tokens, embeddings

def apply_pca(embeddings, n_components=2):
    """Apply PCA to reduce dimensionality."""
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings_scaled)
    
    variance_explained = pca.explained_variance_ratio_
    print(f"[+] PCA variance explained: {variance_explained}")
    print(f"[+] Total variance: {sum(variance_explained):.4f}")
    
    return reduced, pca

def visualize_projection(reduced, tokens, save_path='projection_2d.png'):
    """Create visualization of the 2D projection."""
    plt.figure(figsize=(14, 10))
    
    # Scatter plot
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=100, c='steelblue', edgecolors='black')
    
    # Annotate each point with its token
    for i, token in enumerate(tokens):
        plt.annotate(token, (reduced[i, 0], reduced[i, 1]), 
                    fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Token Embeddings Projected to 2D Space (PCA)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[+] Visualization saved to {save_path}")
    plt.close()

def nearest_neighbor_chain(reduced, tokens, start_idx=None):
    """
    Traverse the 2D space using nearest-neighbor algorithm.
    
    This creates a path through the projection by always moving to the
    closest unvisited point, revealing the hidden flag sequence.
    """
    n = len(tokens)
    visited = [False] * n
    result = []
    
    # Try different starting points if not specified
    if start_idx is None:
        # Look for 'H' as likely start of HTB{ flag
        possible_starts = [i for i, t in enumerate(tokens) if t == 'H']
    else:
        possible_starts = [start_idx]
    
    best_result = None
    
    for start in possible_starts:
        visited = [False] * n
        result = []
        current = start
        
        for _ in range(n):
            visited[current] = True
            result.append(tokens[current])
            
            # Find nearest unvisited neighbor
            min_dist = float('inf')
            nearest = -1
            
            for i in range(n):
                if not visited[i]:
                    dist = np.linalg.norm(reduced[current] - reduced[i])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i
            
            if nearest == -1:
                break
            current = nearest
        
        candidate = ''.join(result)
        
        # Check if this looks like a valid flag
        if candidate.startswith('HTB{') and '}' in candidate:
            # Extract the flag portion
            end_idx = candidate.index('}', 4) + 1
            flag = candidate[:end_idx]
            
            # Basic validation: flag should be reasonable length
            if 15 <= len(flag) <= 50:
                print(f"[+] Found potential flag starting from index {start} ('{tokens[start]}'): {flag}")
                if best_result is None or len(flag) > len(best_result):
                    best_result = flag
    
    return best_result

def main():
    print("=" * 70)
    print("Lost in Hyperspace - Solution")
    print("=" * 70)
    
    # Load the data
    tokens, embeddings = load_data()
    
    # Apply PCA to reduce to 2D
    print("\n[*] Applying PCA dimensionality reduction...")
    reduced_2d, pca = apply_pca(embeddings, n_components=2)
    
    # Visualize the projection
    print("\n[*] Creating visualization...")
    visualize_projection(reduced_2d, tokens)
    
    # Find the flag using nearest neighbor traversal
    print("\n[*] Searching for flag using nearest-neighbor traversal...")
    flag = nearest_neighbor_chain(reduced_2d, tokens)
    
    if flag:
        print("\n" + "=" * 70)
        print(f"FLAG FOUND: {flag}")
        print("=" * 70)
        return flag
    else:
        print("\n[-] No valid flag found. Try manual analysis of the visualization.")
        return None

if __name__ == "__main__":
    flag = main()
