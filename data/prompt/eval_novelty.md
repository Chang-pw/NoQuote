You are an expert evaluator. Given a “context” text and a single “candidate quote,” rate the quote on the dimension below:

**Surprise Novelty (1–5)**: How surprising, clever, or “wow-worthy” is this quote in light of the context?  
(1 = entirely predictable or trivial; 5 = genuinely unexpected yet fitting, highly insightful)

Please output in this YAML format:

novelty:
  reason: brief justification for your novelty score
  score: X

Note:
- If the quote is in Chinese, write the reason in **Chinese**; otherwise, write it in **English**.
- Only evaluate this single dimension.
- Please firstly give the reason and then give the score.

Example1:
Context: "在个人形象这个议题上，传统儒家主张通过修身与道德完善来实现个人的提升。如今，随着互联网的迅猛发展与社交媒体的兴起，人们愈发关注他人眼中的自己。"
Quote: "你的品牌，是别人背后对你说的话。"
Deep Meaning of Quote: "表达了真正的声誉存在于我们无法控制的空间中，体现在他人背后的真实评价。"
Output:
novelty:
  reason: "该引语以现代‘品牌’理念重新诠释了个人形象，令人耳目一新，同时准确捕捉了社交时代中他人评价对自我认知的影响。"
  score: 5

Example2:
Context: "In times of uncertainty and crisis, leaders are expected to provide clarity, calm, and a sense of direction. Their communication style can profoundly shape public morale and trust."
Quote: "A leader is one who knows the way, goes the way, and shows the way."
Deep Meaning of Quote: "Expresses that true leadership is lived through example."
Output:
novelty:
  reason: "This quote is overused and generic—it doesn’t offer a surprising or nuanced insight about leadership in uncertain or crisis conditions. It’s surface-level and predictable."
  score: 2

Context: "<context>"  
Quote: "<quote>"  
Deep Meaning of Quote: "<deepmeaning>"  
Please start your evaluation and provide the output in the specified YAML format without other infomation or strings.