#%%
from grader import *
#%%
text = """🎉 ($)79 - 🎬9 = 🏁$70 (money left after birthday) 🏁$70 - 🏁$32 = 🛒$38 (cost of baseball gear) So, Benny spent 🛒
$38 on baseball gear. 🛒✨ 38"""

grade_output_no_tags(text, "38", 0)
# %%
text = """<reasoning> Alright, let's solve this step by step! ♂️的数量是49个，他给了Tim 13个。我们需要从总数中减去给出去的数量。 49 - 13 = 36个。所以Jason现在有36个贝壳。👍</reasoning><answer>5</answer>"""
grade_output_in_tags(text, "5", 0)

# %%
