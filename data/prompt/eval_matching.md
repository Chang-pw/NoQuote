You are an expert evaluator. Given a “context” text and a single “candidate quote,” rate the quote on the dimension below:

**Semantic Matching (1–5)**: How well does this quote align with the main topic, argument, or intent of the context?  
(1 = off-topic; 5 = directly and indispensably connected)

Please output in this YAML format:

matching:
  reason: brief justification for your matching score
  score: Y

Note:
- If the quote is in Chinese, write the reason in **Chinese**; otherwise, write it in **English**.
- Only evaluate this single dimension.
- Please firstly give the reason and then give the score.

Example1:
Context: "在个人形象这个议题上，传统儒家主张通过修身与道德完善来实现个人的提升。如今，随着互联网的迅猛发展与社交媒体的兴起，人们愈发关注他人眼中的自己。"
Quote: "你的品牌，是别人背后对你说的话。"
Deep Meaning of Quote: "表达了真正的声誉存在于我们无法控制的空间中，体现在他人背后的真实评价。"
Output:
matching:
  reason: "引语与上文关于‘社交媒体时代形象来源于他人看法’的论点高度契合，是一次恰当而深刻的补充。"
  score: 5

Example2:
Context: "In times of uncertainty and crisis, leaders are expected to provide clarity, calm, and a sense of direction. Their communication style can profoundly shape public morale and trust."
Quote: "A leader is one who knows the way, goes the way, and shows the way."
Deep Meaning of Quote: "Expresses that true leadership is lived through example."
Output:
matching:
  reason: "While the quote is broadly about leadership, it lacks specificity to the context of crisis communication or uncertainty. It fits the topic loosely but doesn’t enrich the argument."
  score: 3


Context: "<context>"  
Quote: "<quote>"  
Deep Meaning of Quote: "<deepmeaning>"  

Please start your evaluation and provide the output in the specified YAML format without other infomation or strings.