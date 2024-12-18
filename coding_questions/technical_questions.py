##################################################################################################################################

'''Given a string 'st', calculate the occurrence of each character with loop.'''

st = 'abbcccdddd'

# Initialize an empty dictionary to store character counts
char_count = {}

# Loop through each character in the string
for char in st:
    # If the character is already in the dictionary, increment its count
    if char in char_count:
        char_count[char] += 1
    # If the character is not in the dictionary, add it with count 1
    else:
        char_count[char] = 1

# Print the occurrence of each character
for char, count in char_count.items():
    print(f"Character: {char}, Count: {count}")

##################################################################################################################################

'''Given a list lst = [1, 2, 3,..., 23], create a list of lists of indexes with at most 2 numbers from the given list whose sum
is equal to 24.'''

lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Initialize an empty list to store the result
index_pairs = []

# Iterate through the list and check each pair of numbers
for i in range(len(lst)):
    for j in range(i + 1, len(lst)):
        if lst[i] + lst[j] == 24:
            # Append the pair of indexes to the result list
            index_pairs.append([i, j])

# Print the result
print(index_pairs)

'''With omega log n complexity'''
# Initialize an empty dictionary to store numbers and their indices
index_map = {}

# Initialize a list to store the result of index pairs
index_pairs = []

# Iterate through the list
for i, num in enumerate(lst):
    # Calculate the complement
    complement = 24 - num
    
    # Check if the complement exists in the dictionary
    if complement in index_map:
        # Append the pair of indices to the result list
        index_pairs.append([index_map[complement], i])
    
    # Store the current number and its index in the dictionary
    index_map[num] = i

# Print the result
print(index_pairs)

##################################################################################################################################

'''Write a program to find maximum number of drop points that can be covered by flying over the terrain once. The points are
given as input in the form of integers coordinates in a two dimensional field. The flight path can be horizontal or vertical,
but not a mix of the two or diagonal.'''

def max_drop_points(x_coords, y_coords):
    # Dictionaries to count occurrences of points with the same x and y coordinates
    x_count = {}
    y_count = {}
    
    # Ensure both lists are of the same length
    if len(x_coords) != len(y_coords):
        raise ValueError("x and y coordinates must have the same length")
    
    # Iterate using index, without using zip
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        
        if x in x_count:
            x_count[x] += 1
        else:
            x_count[x] = 1
        
        if y in y_count:
            y_count[y] += 1
        else:
            y_count[y] = 1
    
    # Filter out any groups with only 1 point
    max_x_points = max([count for count in x_count.values() if count > 1], default=0)
    max_y_points = max([count for count in y_count.values() if count > 1], default=0)
    
    # Return the maximum of the two
    return max(max_x_points, max_y_points)

# Example usage:
x_coords = [1, 2, 3, 4, 4, 2, 2]
y_coords = [2, 2, 2, 5, 1, 3, 4]

result = max_drop_points(x_coords, y_coords)
print(f"Maximum number of drop points covered: {result}")

##################################################################################################################################

'''Write a program to find maximum number of chocolates that can be picked from the jars in such a way that the chocolates are
not picked from jars next to each other.'''

def max_chocolates(chocolates):
    n = len(chocolates)
    
    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return chocolates[0]
    
    # Initialize the DP array
    dp = [0] * n
    
    # Set up the first two values
    dp[0] = chocolates[0]
    dp[1] = max(chocolates[0], chocolates[1])
    
    # Fill in the dp array using the recurrence relation
    for i in range(2, n):
        dp[i] = max(chocolates[i] + dp[i-2], dp[i-1])
    
    # The last element of dp array will have the answer
    return dp[-1]

# Example usage:
chocolates = [6, 7, 1, 30, 8, 2, 4]
result = max_chocolates(chocolates)
print(f"Maximum chocolates that can be picked: {result}")

##################################################################################################################################

'''Write a program to rotation a n*m matrix by 90 degrees.'''

def rotate_matrix_90_clockwise(matrix):
    # Get the dimensions of the matrix
    n = len(matrix)        # Number of rows
    m = len(matrix[0])     # Number of columns

    # Create a new matrix to store the rotated version
    rotated_matrix = [[0] * n for _ in range(m)]

    # Fill the rotated matrix by adjusting the positions
    for i in range(n):
        for j in range(m):
            rotated_matrix[j][n-1-i] = matrix[i][j]

    # Store the rotated matrix in a variable
    result_matrix = '\n'.join([' '.join(map(str, row)) for row in rotated_matrix])
    
    # Return the formatted matrix as a string
    return result_matrix

# Example usage:
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Store the rotated matrix in a variable and return it
rotated_matrix = rotate_matrix_90_clockwise(matrix)

# Print the result
print(rotated_matrix)

##################################################################################################################################

'''Write a program to find out the elements which are largest in a row and smallest in a column in a n*m matrix.'''

def find_largest_in_row_and_smallest_in_column(matrix, n, m):
    # Result list to store the saddle points
    result = []

    # Traverse each row
    for i in range(n):
        # Find the largest element in the row and its column index
        row_max = max(matrix[i])
        col_index = matrix[i].index(row_max)

        # Check if this element is the smallest in its column
        is_smallest_in_column = True
        for j in range(n):  # Traverse the column of the row_max element
            if matrix[j][col_index] < row_max:
                is_smallest_in_column = False
                break

        # If the element is the largest in the row and smallest in the column, add it to the result
        if is_smallest_in_column:
            result.append((row_max, i, col_index))  # Add element and its position (row, col)

    # Return the result list if it contains saddle points, otherwise return None
    return result if result else None


# Example usage
n, m = 3, 3
matrix = [
    [3, 8, 9],
    [5, 2, 4],
    [7, 1, 6]
]

result = find_largest_in_row_and_smallest_in_column(matrix, n, m)

if result:
    for val, row, col in result:
        print(f"Element {val} is largest in row {row+1} and smallest in column {col+1}")
else:
    print("No such element found.")

##################################################################################################################################

'''Find the substring from a string that is the same when read forward and backward in a single function and return `None` if no
palindrome substring.'''

def longest_palindrome(s):
    if not s:
        return None

    def expand_around_center(left, right):
        # Expand as long as the substring is a palindrome
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # Return the palindrome substring
        return s[left + 1:right]

    longest_palindrome = None

    for i in range(len(s)):
        # Odd length palindrome (single character center)
        palindrome_odd = expand_around_center(i, i)
        if longest_palindrome is None or len(palindrome_odd) > len(longest_palindrome):
            longest_palindrome = palindrome_odd

        # Even length palindrome (two character center)
        palindrome_even = expand_around_center(i, i + 1)
        if len(palindrome_even) > len(longest_palindrome):
            longest_palindrome = palindrome_even

    # Return None if no palindrome was found, else return the longest palindrome
    return longest_palindrome if longest_palindrome and len(longest_palindrome) > 1 else None

# Example usage
s = "abc"
print(longest_palindrome(s))  # Output: None

s2 = "babad"
print(longest_palindrome(s2))  # Output: "bab" or "aba"

##################################################################################################################################

'''Given a list of integers, find the maximum difference of 2 elements where the larger number appears after the smaller number.'''

def max_difference(arr):
    if len(arr) < 2:
        return 0  # Not enough elements for a valid difference

    min_element = arr[0]
    max_diff = float('-inf')

    for i in range(1, len(arr)):
        diff = arr[i] - min_element
        if diff > max_diff:
            max_diff = diff
        if arr[i] < min_element:
            min_element = arr[i]

    return max_diff if max_diff > 0 else 0  # Return 0 if no valid difference

##################################################################################################################################

'''Convert a non-negative integer num to its English words representation.'''

def numberToWords(self, num: int) -> str:
    if num == 0:
        return "Zero"

    # Words for units, teens, tens, and thousands
    units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    thousands = ["", "Thousand", "Million", "Billion"]

    def helper(n):
        if n == 0:
            return ""
        elif n < 10:
            return units[n]
        elif n < 20:
            return teens[n - 10]
        elif n < 100:
            return tens[n // 10] + ("" if n % 10 == 0 else " " + units[n % 10])
        else:
            return units[n // 100] + " Hundred" + ("" if n % 100 == 0 else " " + helper(n % 100))

    result = ""
    for i, chunk in enumerate(thousands):
        if num % 1000 != 0:
            result = helper(num % 1000) + " " + chunk + ("" if result == "" else " " + result)
        num //= 1000

    return result.strip()

##################################################################################################################################

'''You are given a 2D maze represented by a grid. The maze consists of open paths (0) and walls (1). Your goal is to implement
a function that finds and marks the solution path in the maze using the value 2. The maze solver should start from the top-left
corner (cell [0, 0]) and reach the bottom-right corner (cell [len(maze) - 1, len(maze[0]) - 1]). The solution path can only move
through open paths and should navigate around walls.'''

def solve_maze(maze):
    rows, cols = len(maze), len(maze[0])
    solution_path = []

    # Helper function to perform DFS
    def dfs(x, y):
        # Base case: reached the bottom-right corner
        if x == rows - 1 and y == cols - 1:
            maze[x][y] = 2
            solution_path.append((x, y))
            return True

        # Check if the current cell is out of bounds or a wall
        if x < 0 or y < 0 or x >= rows or y >= cols or maze[x][y] != 0:
            return False

        # Mark the cell as part of the solution path
        maze[x][y] = 2
        solution_path.append((x, y))

        # Explore neighbors in the order: Down, Right, Up, Left
        if (
            dfs(x + 1, y) or  # Down
            dfs(x, y + 1) or  # Right
            dfs(x - 1, y) or  # Up
            dfs(x, y - 1)     # Left
        ):
            return True

        # If no path is found, backtrack
        maze[x][y] = 0
        solution_path.pop()
        return False

    if dfs(0, 0):
        return maze, solution_path
    else:
        return None, []  # No solution

# Hardcoded maze
hardcoded_maze = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
]

# Solve the maze
solved_maze, path = solve_maze(hardcoded_maze)

if solved_maze:
    print("Solved Maze:")
    for row in solved_maze:
        print(row)
    print("\nSolution Path:", path)
else:
    print("No solution found.")

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
