# app.py
from flask import Flask, render_template, request, jsonify
import random
import time
import plotly.graph_objects as go
import json
import io

 

app = Flask(__name__)
def generate_random_array(size: int) -> list[int]:
    """
    Generates a random array of integers between 1 and 1000 of the specified size.
    
    Args:
        size (int): The size of the array to generate.
        
    Returns:
        list[int]: A list of random integers.
    """
    return [random.randint(1, 1000) for _ in range(size)]


def insertion_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the insertion sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    # Check if the array is already sorted
    if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
        return arr
    
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        
        # Move elements of arr[0..i-1], that are greater than key, to one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        # Place the key at the correct position
        arr[j + 1] = key
    
    return arr

def bubble_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the optimized bubble sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    n = len(arr)

    # Early exit if the array is already sorted
    if all(arr[i] <= arr[i + 1] for i in range(n - 1)):
        return arr

    for i in range(n):
        swapped = False
        # Track the last swapped index to avoid unnecessary comparisons
        last_swapped = n - 1
        for j in range(0, last_swapped - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
                last_swapped = j  # Update the last swapped index

        # If no elements were swapped, the array is already sorted
        if not swapped:
            break

    return arr

def selection_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the selection sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    n = len(arr)

    # Early exit if the array is already sorted
    if all(arr[i] <= arr[i + 1] for i in range(n - 1)):
        return arr

    for i in range(n):
        # Find the minimum element in the unsorted portion of the array
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        
        # Swap only if the minimum index has changed
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]

    return arr

def merge_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the merge sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    # Base case: If array has only one element, it's already sorted
    if len(arr) <= 1:
        return arr

    # Check if the array is already sorted
    if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
        return arr

    # Split the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursive calls to sort both halves
    merge_sort(left_half)
    merge_sort(right_half)

    i = j = k = 0

    # Merging the sorted halves
    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    # Copy any remaining elements from the left half
    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1

    # Copy any remaining elements from the right half
    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

    return arr

def quick_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the quick sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    def _quick_sort(arr, low, high):
        if low < high:
            # Partition the array and get the pivot index
            pi = partition(arr, low, high)
            
            # Recursively sort the left and right subarrays
            _quick_sort(arr, low, pi - 1)
            _quick_sort(arr, pi + 1, high)

    def partition(arr, low, high):
        pivot = arr[high]  # Taking the last element as the pivot
        i = low - 1  # Index of the smaller element

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]  # Swap

        arr[i + 1], arr[high] = arr[high], arr[i + 1]  # Swap pivot to its correct place
        return i + 1

    # Call the helper function with initial low and high values
    _quick_sort(arr, 0, len(arr) - 1)
    return arr

def heapify(arr: list[int], n: int, i: int) -> None:
    """
    Ensures the subtree rooted at index i follows the heap property.
    
    Args:
        arr (list[int]): The array representing the heap.
        n (int): The size of the heap.
        i (int): The index of the root element of the subtree.
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1  # Left child index
    right = 2 * i + 2  # Right child index

    # Check if left child exists and is greater than the root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # Check if right child exists and is greater than the largest found so far
    if right < n and arr[right] > arr[largest]:
        largest = right

    # If the largest is not the root, swap and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        heapify(arr, n, largest)  # Recursively heapify the affected subtree

def heap_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the heap sort algorithm.
    
    Args:
        arr (list[int]): The array to be sorted.
        
    Returns:
        list[int]: The sorted array.
    """
    n = len(arr)

    # Early exit for empty array
    if n == 0:
        return arr

    # Build a max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements from the heap
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # Move current root to end
        heapify(arr, i, 0)  # Call heapify on the reduced heap

    return arr

def counting_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the counting sort algorithm.

    Args:
        arr (list[int]): The array to be sorted.

    Returns:
        list[int]: The sorted array.
    """
    # Early exit for empty array
    if not arr:
        return arr

    max_val = max(arr)
    min_val = min(arr)
    range_of_elements = max_val - min_val + 1

    # Create count array and initialize it to 0
    count = [0] * range_of_elements

    # Count the occurrences of each element
    for num in arr:
        count[num - min_val] += 1

    # Update count array to store cumulative counts
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Create a sorted output array
    output = [0] * len(arr)

    # Build the output array in reverse order to maintain stability
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    # Copy the sorted output array back to the original array
    for i in range(len(arr)):
        arr[i] = output[i]

    return arr

def counting_sort_for_radix(arr: list[int], exp: int) -> None:
    """
    A stable counting sort function that sorts the array based on the digit represented by exp.

    Args:
        arr (list[int]): The array to be sorted.
        exp (int): The exponent representing the digit place (1 for units, 10 for tens, etc.).
    """
    n = len(arr)
    output = [0] * n  # Output array to store sorted values
    count = [0] * 10  # Count array for digits (0-9)

    # Count occurrences of each digit
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1

    # Update count array to store cumulative counts
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array in reverse order to maintain stability
    for i in range(n - 1, -1, -1):
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1

    # Copy the output array back to the original array
    for i in range(n):
        arr[i] = output[i]


def radix_sort(arr: list[int]) -> list[int]:
    """
    Sorts an array using the radix sort algorithm.

    Args:
        arr (list[int]): The array to be sorted.

    Returns:
        list[int]: The sorted array.
    """
    # Early exit for empty array
    if not arr:
        return arr

    max_val = max(arr)  # Find the maximum value to know the number of digits
    exp = 1  # Start with the least significant digit

    # Perform counting sort for each digit
    while max_val // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10  # Move to the next significant digit

    return arr

def bucket_sort(arr):
    if len(arr) == 0:
        return arr

    # Create n empty buckets
    bucket_count = len(arr)
    buckets = [[] for _ in range(bucket_count)]

    # Put array elements in different buckets
    for num in arr:
        # Ensure that 1.0 falls into the last bucket
        index = min(int(num * bucket_count), bucket_count - 1)
        buckets[index].append(num)

    # Sort individual buckets
    for bucket in buckets:
        bucket.sort()

    # Concatenate all sorted buckets
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr




# Array sizes
array_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

# Sorting algorithms
sorting_algorithms = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Heap Sort": heap_sort,
    "Counting Sort": counting_sort,
    "Radix Sort": radix_sort,
    "Bucket Sort": bucket_sort
}

@app.route('/')
def index():
    """
    Renders the main index page.
    """
    return render_template('index.html')


@app.route('/run_algorithm', methods=['POST'])
def run_algorithm():
    """
    Endpoint to run the sorting algorithm and return its execution time.
    """
    algorithm_name = request.form['algorithm']
    array_size = int(request.form['array_size'])

    random_array = generate_random_array(array_size)
    algorithm = sorting_algorithms[algorithm_name]

    start_time = time.time()
    algorithm(random_array.copy())
    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return jsonify({
        'algorithm': algorithm_name,
        'array_size': array_size,
        'time': execution_time
    })


@app.route('/plot_results', methods=['POST'])
def plot_results():
    results = request.json
    print(results)  # Debugging: Check if the results format is as expected

    fig = go.Figure()

    # Iterate through each sorting algorithm's results
    for algorithm, data in results.items():
        # Extract array sizes and corresponding times from the results
        sizes = [point['array_size'] for point in data]
        times = [point['time'] for point in data]

        # Add a trace for each algorithm
        fig.add_trace(go.Scatter(x=sizes, y=times, mode='lines+markers', name=algorithm))

    # Determine the maximum array size for setting the x-axis range
    max_size = max(point['array_size'] for data in results.values() for point in data)

    # Update the layout of the figure
    fig.update_layout(
        title="Empirical Time Complexity of Sorting Algorithms",
        xaxis_title="Array Size",
        yaxis_title="Time (ms)",  # Y-axis for time taken in milliseconds
        legend_title="Sorting Algorithms",
        hovermode="x unified",
        template="plotly_white",
        xaxis=dict(range=[1, max_size])  # Set x-axis range from 1 to max input array size
    )

    # Return the JSON representation of the figure
    return jsonify(fig.to_json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)