from .utils import label_extract
from .rag.rag_retrieval import dm_retrieval
from .context_summary import context_summary
from .rag.hard_filter import hard_label_filter

def hard_retrieval(quotes, label_list, threshold=0.6):
    core_domains, core_insights, _, _, _, _, _ = label_extract(label_list)
    insights = [quote["core_insights"] for quote in quotes]
    insights_bool_list = hard_label_filter(insights, core_insights, threshold=threshold)

    filtered_quotes = []
    aa = []
    for quote_item, keep in zip(quotes, insights_bool_list):
        if keep:
            filtered_quotes.append(quote_item)
        else:
            aa.append(quote_item)

    quotes = [i["quote"] for i in filtered_quotes]
    aa_quotes = [i["quote"] for i in aa]
    return filtered_quotes, quotes, aa, aa_quotes


def reasonable_retrieval(context, top_n, threshold, fixed_info=None, mode="ch"):
    fixed_info = fixed_info or {}
    if fixed_info:
        sm_context = fixed_info["summary_context"]
        dm_context = [fixed_info["dm_context"]]
        label_list = fixed_info["label_list"]
    else:
        sm_context, dm_context, label_list = context_summary(context)

    dm_filter_result, dm_filter_quotes = dm_retrieval(dm_context, top_n=top_n)
    hd_filter_result, hd_filter_quotes, aa_result, aa_quotes = hard_retrieval(
        dm_filter_result, label_list, threshold=threshold
    )
    return hd_filter_quotes, sm_context, dm_filter_quotes, dm_context[0], label_list

