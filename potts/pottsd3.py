# Dependencies:
#   sagemath, tested with version >= 10.2
#   tqdm, installed below
#   wolfram engine with the wolframscript command in the PATH

# %% Install tqdm through pip
# subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

# %% Imports
# from tqdm.autonotebook import tqdm

from sage.symbolic.expression import Expression
from sage.rings.rational import Rational
from sage.rings.integer import Integer

from dataclasses import dataclass
from typing import List

load("potts_occ.py")
b = var("b", latex_name="\\beta")

Ls = list(gen_local_views(2,2))

# L = Ls[0]
# print('Studying local view:')
# L.show()
# for cw, m, sigma in L.gen_all_spin_assignments():
#     print(sigma, cw*exp(-b*m))
#     L.show(sigma)


L = Ls[-1]
print('Studying local view:')
L.show()
for cw, m, sigma in L.gen_all_spin_assignments():
    print(sigma.values(), cw, exp(-b*m))
    # L.show(sigma)

z = 0
for cw, m, sigma in L.gen_all_spin_assignments():
    z += cw*exp(-b*m)
print(simplify(z))


# %%
