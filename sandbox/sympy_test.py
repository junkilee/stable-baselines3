from sympy import symbols
from sympy.logic.boolalg import to_cnf
from sympy.logic import simplify_logic

a, b, c = symbols('a,b,c')
print(to_cnf(a >> b))