import numpy as np
from collections import Counter
import heapq

def calculate_entropy(image_array):
    """Calculate the entropy of a grayscale image."""
    # histogram to get freq of each pixel values
    histogram_arr = Counter(image_array.flatten())
    total_pixel_sum = sum(histogram_arr.values())

    probabilities = {
        i: val / total_pixel_sum for i, val in histogram_arr.items()
    }

    entropy = -sum(prob * np.log2(prob) for prob in probabilities.values())
    return entropy

def generate_huffman_tree(histogram_arr):
    # Create a priority queue with initial frequencies from the image histogram.
    # Each entry in the queue is a tuple: (frequency, [pixel_value, code_string])
    priority_queue = [[freq, [pixel_val, ""]] for pixel_val, freq in histogram_arr.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        # Pop the two nodes with the lowest frequency
        low_freq_node = heapq.heappop(priority_queue)
        high_freq_node = heapq.heappop(priority_queue)

        # Append '0' to the code of the lower frequency node,
        # and '1' to the code of the higher frequency node.
        for pair in low_freq_node[1:]:
            pair[1] = '0' + pair[1]
        for pair in high_freq_node[1:]:
            pair[1] = '1' + pair[1]

        # Merge the two nodes and push back onto the priority queue
        heapq.heappush(priority_queue, [low_freq_node[0] + high_freq_node[0]] + low_freq_node[1:] + high_freq_node[1:])
    
    # The final node is the root of the Huffman tree
    return priority_queue[0]

def calculate_avg_code_length(huffman_tree, histogram_arr):
    # Dictionary to hold the Huffman codes for each pixel value
    huffman_codes = {}

    # Helper function to recursively generate Huffman codes
    def generate_huffman_codes(node, code_prefix=""):
        # Base case: If the node is a leaf, assign its code
        if len(node) == 2:
            pixel_val, _ = node
            huffman_codes[pixel_val] = code_prefix
        else:
            # Recursive case: Traverse left and right children
            generate_huffman_codes(node[1], code_prefix + "0")
            generate_huffman_codes(node[2], code_prefix + "1")

    generate_huffman_codes(huffman_tree)

    # Calculate the total number of pixels in the image
    total_pixel_sum = sum(histogram_arr.values())

    # Calculate the average code length weighted by the frequency of each pixel value
    avg_code_length = sum(len(code) * (freq / total_pixel_sum) for pixel_val, code in huffman_codes.items() for freq in histogram_arr.values() if pixel_val in histogram_arr)
    return avg_code_length

def get_avg_code_length(image_array):
    """Calculate the average code length of an image."""
    # histogram to get freq of each pixel values
    histogram_arr = Counter(image_array.flatten())
    huffman_tree = generate_huffman_tree(histogram_arr)
    avg_code_length = calculate_avg_code_length(huffman_tree, histogram_arr)
    return avg_code_length

