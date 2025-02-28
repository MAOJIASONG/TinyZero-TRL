ORM_PROMPT = """You are an expert in verifying if two math answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are mathematically equivalent.
Your task is to determine if two mathematical answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical mathematical values or expressions, even when written in different forms or notations.

Guidelines for equivalence:
- Different forms of the same number (e.g., 0.5 = 1/2 = 50%)
- Algebraically equivalent expressions (e.g., (x+1)^2 = x^2 + 2x + 1)
- Geometrically equivalent expressions (e.g., r²π = πr²)
- Trigonometrically equivalent expressions (e.g., sin²θ + cos²θ = 1)
- Semantic equivalence (e.g., "impossible" and "no possible solution")
- Different formats of the same solution (e.g., (1,1,1,3) and a=1,b=1,c=1,p=3)
- Solutions with different or no units (e.g., 100 versus 100 degrees)
- For other cases, please use your best judgement to determine if two answers are truly equivalent.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]

-----
Examples:
Problem: What is the area of a circle with radius 2?
Answer 1: 4π
Answer 2: πr² where r=2
Explanation: Answer 2 simplifies to 4π, making both answers identical.
[[YES]]

Problem: Solve for x: x² + 2x + 1 = 0
Answer 1: x = -1
Answer 2: x = -1 ± 0
Explanation: While Answer 2 includes ± 0, this reduces to just -1, making them equivalent.
[[YES]]

Problem: Find all positive integers $a,b,c$ and prime $p$ satisfying that\n\\[ 2^a p^b=(p+2)^c+1.\\]
Answer 1: a=1, b=1, c=1, p=3
Answer 2:  (1, 1, 1, 3)
Explanation: Both answers represent exactly the same solution, just written in different formats. Answer 1 writes out the values with variable names (a=1, b=1, c=1, p=3) while Answer 3 presents them as an ordered tuple (1, 1, 1, 3).
[[YES]]

Problem: The sides of a $99$ -gon are initially colored so that consecutive sides are red, blue, red, blue,..., red, blue, yellow. We make a sequence of modifications in the coloring, changing the color of one side at a time to one of the three given colors (red, blue, yellow), under the constraint that no two adjacent sides may be the same color. By making a sequence of such modifications, is it possible to arrive at the coloring in which consecutive sides \nare red, blue, red, blue, red, blue,..., red, yellow, blue?
Answer 1: There is no such coloring.
Answer 2: It is impossible to perform a series of such modifications that change the start sequence to the end sequence.
Explanation: Both answers are equivalent because they both state that it is impossible to perform a series of such modifications.
[[YES]]

Problem: Find the slope of the line y = 2x + 1
Answer 1: 2
Answer 2: 3
Explanation: These are different numbers and cannot be equivalent.
[[NO]]
-----
"""