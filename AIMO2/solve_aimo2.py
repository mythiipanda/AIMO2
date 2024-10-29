import csv
import re

from IPython import get_ipython
from tqdm.auto import tqdm

def solve_problem(id, problem):
    # Prompt engineering
    sampling_params = vllm.SamplingParams(
      temperature=0.00,
      max_tokens=1024
    )
    prompt = f"Let's solve the following AIMO2 problem step by step:\n{problem}\n\nStep 1: "

    # Generate solution using LLM
    solution_steps = llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)

    # Extract Python code from solution steps
    python_code = extract_python_code(solution_steps)

    # Run Python code using PythonREPL
    result = PythonREPL.run(python_code)

    return result

def extract_python_code(solution_steps):
    # Extract Python code from the generated solution steps
    # Implement this function based on the format of the generated solution
    # Return the extracted Python code
    
    # Assuming the Python code is enclosed in ```python and ``` delimiters
    python_code = re.search(r"```python(.*?)```", solution_steps, re.DOTALL)
    
    if python_code:
        return python_code.group(1).strip()
    else:
        return None

def main():
    with open('reference.csv', 'r') as file:
        reader = csv.DictReader(file)
        answers = []

        for row in reader:
            id = row['id']
            problem = row['problem']

            print(f"Solving problem {id}:")
            print(problem)

            solution = solve_problem(id, problem)

            print(f"Your solution: {solution}")
            print()

            answers.append({'id': id, 'solution': solution})

        # Save answers to a CSV file
        with open('answers.csv', 'w') as outfile:
            fieldnames = ['id', 'solution']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(answers)

if __name__ == '__main__':
    # Initialize LLM and Tokenizer
    llm = vllm.LLM(
        "Qwen/Qwen2.5-Math-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        dtype="half",
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Initialize PythonREPL
    ipython = get_ipython()
    PythonREPL = ipython.run_cell_magic("capture", "--no-stderr", "from IPython import InteractiveShell; PythonREPL = InteractiveShell.instance(); PythonREPL.run_cell('%load_ext autoreload'); PythonREPL.run_cell('%autoreload 2')")

    main()
