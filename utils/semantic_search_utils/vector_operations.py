def add_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return [a + b for a, b in zip(v1, v2)]

def subtract_vectors(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return [a - b for a, b in zip(v1, v2)]

def dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    return sum(a * b for a, b in zip(v1, v2))

def euclidean_norm(vec):
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5

def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same length")
    
    dot_product_result = dot_product(v1, v2)
    norm_v1 = euclidean_norm(v1)
    norm_v2 = euclidean_norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product_result / (norm_v1 * norm_v2)


if __name__ == "__main__":
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    print("Addition of v1 and v2:", add_vectors(v1, v2))
    print("Subtraction of v1 and v2:", subtract_vectors(v1, v2))
    print("Dot product of v1 and v2:", dot_product(v1, v2))
    print("Euclidean norm of v1:", euclidean_norm(v1))
    print("Euclidean norm of v2:", euclidean_norm(v2))
    print("Cosine similarity of v1 and v2:", cosine_similarity(v1, v2))