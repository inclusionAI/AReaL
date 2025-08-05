from realhf.impl.dataset.math_parser_greso import process_results_greso
from math_verify import parse, verify

# completions = """To find $g(2)$, let's multiply both sides of the equation by $g(x)$ to clear the fraction on the left and isolate $g(2)$:

# \[(2^{2^{2008}-1}-1)g(x) = 2^{2^{2007}}\]

# Now, we separate $2^{2^{2008}-1}$ and $2^{2^{2007}}$ into terms that match the right-hand side of the equation. The right-hand side has $(x+1)(x^2+1)(x^4+1)\cdots(x^{2^{2007}}+1)$. This product includes $2^{2^k}$ for $k = 0, 1, 2, \ldots, 2^{2008}-1$.

# Notice that $2^{2^k} + 2^{2^k} = 2^{2^{k+1}}$. Thus, the right-hand side becomes:

# \[= 2^{2^{2007}}\cdot 2^{2^0}\cdot 2^{2^1} \cdot \cdots \cdot 2^{2^{2007-1}}\]

# Simplifying, we get:

# \[= 2^{2^{2007} + 1 - 1}\]

# Since $2^{2^{2008}-1}$ equals $2^{2^{2007} + 1 - 1}$, we can equate the left side to the right side:

# \[(2^{2^{2008}-1}-1)g(2) = 2^{2^{2007} + 1 - 1}\]

# Solving for $g(2)$, we have:

# \[g(2) = \frac{2^{2^{2007} + 1 - 1}}{2^{2^{2008}-1}-1}\]

# \[g(2) = 2^{2^{2007}}\]

# Therefore, $g(2)=2^{2^{2007}}$. The answer is:

# \boxed{2^{2^{2007}}}.<|im_end|>"""

completions = """ First, we substitute $x=y$ into the given equation to get $f(0) = 0$.
Next, we substitute $x=1$ and $y=0$ into the given equation to get $f(1) = (1-0)(f(1) + f(0)) = 1f(1) = 1$.
So we have $f(1) = 1$ and $f(0) = 0$.
Now, we substitute $x=2$ and $y=1$ into the given equation to get $f(3) = (2-1) ((2-1) f(1) + f(1)) = 1(1+1) = 2$.
So we have $f(3) = 2$.
Therefore, the number of possible values of $f(2)$ is 2 and the sum of all possible values of $f(2)$ is $2+2=4$.
So $n \times s = 2 \times 4 = 8$.
The answer is $\\boxed{8}$.<|im_end|>"""

solution = """Setting $x = y,$ we get $f(0) = 0.$

Setting $x = -1$ and $y = 0,$ we get
\[f(1) = -f(-1),\]so $f(-1) = -1.$

Setting $y = 1$ and $y = -1,$ we get
\begin{align*}
f(x^2 - 1) &= (x - 1) (f(x) + 1), \\
f(x^2 - 1) &= (x + 1) (f(x) - 1),
\end{align*}respectively. Hence, $(x - 1) (f(x) + 1) = (x + 1) (f(x) - 1),$ which simplifies to $f(x) = x.$ We can check that this function works. Therefore, $n = 1$ and $s = 2,$ so $n \\times s = \\boxed{2}.$"""


print("--- arealite verify")
print(process_results_greso(completions, solution))



print("--- hf math verify")
print("parse gold")
gold = parse(solution)
print("parse answer")
answer = parse(completions)
print("verify")
print(verify(gold, answer), answer, gold)

print("=== v2: remove <|im_end|> ===")
completions = completions.replace("<|im_end|>", "")

print("--- arealite verify")
print(process_results_greso(completions, solution))


print("--- hf math verify")
print("parse gold")
gold = parse(solution)
print("parse answer")
answer = parse(completions)
print("verify")
print(verify(gold, answer), answer, gold)