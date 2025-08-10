Title: BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent

authors: \author{
Zijian Chen\thanks{Equal Contribution.\quad \faEnvelopeO ~Correspondence: x93ma@uwaterloo.ca}~~$^{\text{,}1}$,
Xueguang Ma$^{*\text{,~\faEnvelopeO}}$$^{\text{,}1}$,
Shengyao Zhuang$^*$$^{\text{,}2\text{,}5}$,
Ping Nie$^3$,
Kai Zou$^3$,\\
\textbf{
Andrew Liu$^1$,
Joshua Green$^1$,
Kshama Patel$^1$,
Ruoxi Meng$^1$,
Mingyi Su$^1$,}\\
\textbf{
Sahel Sharifymoghaddam$^1$,
Yanxi Li$^1$,
Haoran Hong$^1$,
Xinyu Shi$^1$,
Xuye Liu$^1$,}\\
\textbf{
Nandan Thakur$^1$,
Crystina Zhang$^1$,
Luyu Gao$^4$,
Wenhu Chen$^1$,
Jimmy Lin$^1$}
\\
[1ex]
$^1$University of Waterloo,\quad
$^2$CSIRO,\quad
$^3$Independent,\\
$^4$Carnegie Mellon University,
$^5$The University of Queensland,\quad
\\
[1.5ex]
\url{https://texttron.github.io/BrowseComp-Plus/}
}

Insert teaser.png here

caption: Accuracy vs. number of search calls for Deep-Research agents with different retrievers.
The figure shows that **Deep-Research agents mostly improve the final accuracy at a cost of more search calls**, whereas better retrieval systems not only improve the overall accuracy but also reduce the number of search calls. That is, **better retrievers lead to both efficiency and effectiveness.**


## What is this?

**Deep-Research** agents, which integrate large language models (LLMs) with search tools, have shown success in improving the effectiveness of handling complex queries that require iterative search planning and reasoning over search results.

BrowseComp-Plus is the first benchmark for evaluating **retrieval-agent interactions** in Deep-Research, isolating the effect of the retriever and the LLM agent to enable **fair, transparent comparisons of Deep-Research agents**. The benchmark sources challenging, reasoning-intensive queries from OpenAI's [BrowseComp](https://openai.com/index/browsecomp). However, instead of searching the live web, BrowseComp-Plus evaluates against a fixed, curated corpus of ~100K web documents from the web. The corpus includes both human-verified evidence documents sufficient to answer the queries, and mined hard negatives to keep the task challenging.

## Why a fixed corpus?

Existing benchmarks for Deep-Research agents consist of question-answer pairs, and require agents to answer them using live web search APIs in real time. This setup introduces major fairness and reproducibility issues:

1. **The internet is a moving target**. The web constantly changes; thus, a system evaluated today on the web may be able to answer different queries evaluated tomorrow. This is especially crucial in the presence of data leakage (e.g., public releases of query-answer pairs on Hugging Face), which renders evaluations meaningless if agents see leaked data.
2. **Web search APIs lack transparency**. Black-box web search APIs add to the complexity of the moving target; they vary in retrieval algorithms and indexed content overtime, hindering apples-to-apples comparisons across time, even when using the same API.

A fixed corpus gives complete control over the retrieval process used by Deep-Research agents, **isolating the effect of the retriever**. This not only enables fair, reproducible evaluations in the same retrieval setting, but also allows us to systematically compare the effects of different retrievers paired with the same LLM agent, answering the question of **"how much does the retriever matter in Deep-Research?"**

## Dataset Construction

BrowseComp-Plus contains 830 queries sourced from [BrowseComp](https://openai.com/index/browsecomp), each of which could take a human more than 2 hours to answer using a search engine. We carefully construct a corpus of ~100K web documents for these queries, designed to meet three criteria:
1. **Comprehensive Coverage**: The corpus provides complete evidence to support the entire reasoning chain required to answer each question.
2. **Clear Differentiation of Effectiveness**: The corpus contains sufficiently distracting hard negatives to maintain difficulty, capable of distinguishing the effectiveness of various strong Deep-Research agents.
3. **Practical Size**: At a size of 100K, the corpus is large enough to yield reliable research insights, while being computationally reasonable for research purposes.

For each query, we collect the evidence documents in a two-stage process: (1) OpenAI's o3 retrieves candidate evidence documents from the web using the ground-truth question–answer pairs; (2) Human annotators verify the candidates and add missing documents to ensure the corpus contains all evidence needed to fully answer each query.

![positive_collection](positive_collection.png)

In addition to **evidence** documents, annotators also label the documents that _semantically_ contains the final answer, designated as **gold** documents. These labels are later used to perform retriever-only evaluation.
> For example, a query might ask for the number of publications by an author, with the ground-truth answer being "7". A gold document could be the author's personal webpage listing their publications; while it may not contain the string "7" explicitly, it semantically contains the answer.

For the negative collection, we take each query from BrowseComp, and prompt GPT-4o to decompose the query into simpler, self-contained sub-queries. For each sub-query, we use a Google Search API provider to search the web, and scrape the results as hard negatives.

![negative_collection](negative_collection.png)

## Results

![main_table](main_table.png)

We evaluate popular Deep-Research agents paired with different retrievers on the following metrics:

1. **Accuracy**: The percentage of queries answered correctly, judged by gpt-4.1.
2. **Recall**: Fraction of evidence documents retrieved in at least one search call by the agent, relative to all evidence documents.
3. **Search Calls**: Number of search calls issued by the agent.
4. **Calibration Error**: Following BrowseComp, we prompt agents to estimate the confidence of their answers; calibration error measures the gap between a model's predicted confidence and actual accuracy.

![main_table](main_table.png) 

**Effect of Retrievers**
Stronger retrievers (e.g., Qwen3-Embedding-8B) consistently improve end-to-end accuracy of Deep-Research agents. They also reduce the number of search calls, likely because higher-quality initial retrievals reduce the need for follow-up searches; further, fewer search calls translate to fewer output tokens. That is, **better retrievers deliver both efficiency and effectiveness gains.**

**Search Calls vs. Accuracy**
In general, more search calls correlate with higher accuracy. Closed-source agents tend to make substantially more search calls than open-source ones; for instance, OpenAI's gpt-5 and o3 average over 20 search calls per query, while Qwen3-32B and SearchR1-32B make fewer than 2, despite being explicitly prompted to use the tool. This gap in the ability to interleave extensive search calls and reasoning likely contributes to the gap in end-to-end accuracy between closed- and open-source agents.

**Impact of Reasoning Effort**

[reasoning_effort.png]

We analyze how the reasoning effort of LLMs influences answer quality and retrieval behavior. 
To isolate this effect, we focus on the gpt-oss family, which offers three reasoning modes: _low_, _medium_, and _high_. 
These modes differ in the amount of computational effort and deliberation the model applies before producing an answer, with higher modes generally involving longer intermediate reasoning steps. Across all model sizes and retrievers, increasing the reasoning effort consistently boosts accuracy.

**Retriever-only Evaluation**

![retriever_only](retriever_only.png)

We also evaluate the effectiveness of different retrievers alone, measuring each retriever's recall@k and nDCG@k scores against the labeled evidence and gold documents.

- Compared to BM25, Qwen3-Embedding-8B and ReasonIR-8B achieve substantially higher recall and nDCG for both evidence document retrieval and gold document retrieval. 
- We observe a model size scaling law within the Qwen3 embedding family; larger models consistently perform better, with Qwen3-8B surpassing ReasonIR-8B at the 8B scale. 
- However, even the best retriever, Qwen3-Embedding-8B, only achieves 20.3 nDCG@10, showcasing a substantial headroom for improvement.

