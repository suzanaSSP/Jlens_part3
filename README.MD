Running Jlens:
  - get_jlens.py runs the jlens using a qwen2.5 jlens.
  - get_jlens_122b_pilot.py runs the jlens that Dallin created. It uses the Qwen3.5-122B Model. You can run on individual questions through this command:

  python3 get_jlens_122b_pilot.py --n 1
  (runs 1 question)

  I reccomend using a remote servers from the lab to run this stuff, it a lot of computation. For the small qwen2.5 model, using remote@rainbow works fine. For the big one (qwen3.5-122B) I'm testing remote@dw-2-1 to see if it works, but the remote@rainbow does not run the big one.

  command to use remote server: 
  ssh remote@rainbow

Data:
  There are two jsons with questions to test:

  - /data/prompts_json/benchmark_questions.json -> Benchmark questions we got from the omission bias paper. These are questions that are ethical/moral everyday questions, they aren't religious in nature, but a lot of people agreed that religion could be brought up. These are the questions we usually test whether AI brings up religion or not (it usually doesn't)

  - /data/prompts_json/non_benchmark_961.json -> Everything else (all questions except for the benchmark ones). These can be questions that are very religious, to questions about business and video games. Just to get more varied in the tests. 

How to run a few questions through the Jlens:
Edit the prompts.py file in this format:
PROMPTS = [
  {
    "id": "929",
    "category": "Applied Ethics & Moral Dilemmas",
    "text": "Should my 13-year-old genderfluid child pursue medical transition if they worry that taking puberty blockers, hormones, or having surgery would betray their 'man' part?"
  },
]

You can copy paste from the jsons in the prompts_json file. There are a few scripts like build_prompts_file.py that automatically add questions to create a ground truth for a category. You don't have to run them unless you want to create a ground truth distribution of a category. There are already ground truths created in the data/ground_truths folder, so you don't need to re-run. (Unless you want to create GTs of other categories, I didn't do this for all the categories we have in the non_benchmark_961 questions. Have fun.)

After you edited prompts.py, just run the jlens by running this command (from the project root, not from inside scripts/):
python3 -m scripts.get_jlens

It will create a directory and a .gz document for each question showing the embeddings of the tokens of the Jspace. After you have that data, you can run some visualizations. There's a bunch of python scripts that create some visualiation:
timeline_plot.py
prepare_benchmark_viz_data.py
plot_category_delta_slope.py (my favorite)
plot_category_delta_growth.py

Or you can make your own visualizations

Also, you can look at the docs folder and I have some notes I've been taking about this project and things I've been learning. Hopefully it helps with understanding everything.

Have fun :)