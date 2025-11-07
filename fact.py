def factorial(n):
    """Calculate the factorial of a non-negative integer n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def fibonacci(n):
    """Calculate the nth Fibonacci number (0-indexed)."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Example usage
if __name__ == "__main__":
    num = int(input("Enter a number for factorial: "))
    print(f"Factorial of {num} is {factorial(num)}")
    fib_num = int(input("Enter a number for Fibonacci: "))
    print(f"Fibonacci number at position {fib_num} is {fibonacci(fib_num)}")
