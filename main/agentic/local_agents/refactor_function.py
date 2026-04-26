from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

code = """
def f(x):
    return [i*2 for i in x if i%2==0]
"""

prompt = """
Refactor this code:
- improve readability
- add comments
"""

def llm(prompt):
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = [{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

new_code = llm(prompt + code)
print(new_code)

"""
Here's the refactored code with improved readability and comments added:

def double_even_numbers(numbers):

    # Given a list of integers, return a new list containing
    # the double of each even number in the original list.

    # :param numbers: List of integers
    # :return: List of integers (each even number doubled)
    
    doubled_evens = []
    for number in numbers:
        if number % 2 == 0:  # Check if the number is even
            doubled_evens.append(number * 2)
    return doubled_evens

**Explanation:**
- Renamed the function and parameter to more descriptive names.
- Added a docstring describing the function's purpose and parameters.
- Used a standard for-loop with explicit conditions for clarity.
- Added an inline comment inside the loop for the even check.
"""