#%%
from grader import *
#%%
text = "<reasoning>foo</reasoning><answer>;alskdjf;lksadj</answer>"
reasoning, answer = parse_reasoning_and_answer(text)
# %%
reasoning
# %%
answer
# %%
grade_output(reasoning, answer, 1)
# %%
