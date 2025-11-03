import arxiv
from datetime import datetime, timedelta, timezone

def fetch_new_papers(query="cat:cs.AR OR cat:cs.LG", max_results=50, days_back=7):
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    query = ""
    query = "cat:cs.AR OR cs.DC OR cs.ET"
    '''
    category_len = len(computer_science_categories)
    index = 0
    for field in computer_science_categories:
        query += f"cat:cs.{field}"
        if index < category_len - 1:
            query += " OR "
        index += 1
    '''
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    papers = []
    try:
            for result in search.results():
                if result.published >= start_date:
                    papers.append({
                        'title': result.title,
                        'abstract': result.summary,
                        'link': result.entry_id,
                        'pdf': result.pdf_url,
                        'published': result.published
                    })
    except arxiv.UnexpectedEmptyPageError:
        pass
    return papers

# https://arxiv.org/category_taxonomy
computer_science_categories = \
[
    "AI",
    "AR",
    "CC",
    "CE",
    "CG",
    "CL",
    "CR",
    "CV",
    "CY",
    "DB",
    "DC",
    "DL",
    "DM",
    "DS",
    "ET",
    "FL",
    "GL",
    "GR",
    "GT",
    "GR",
    "GT",
    "HC",
    "IR",
    "IT",
    "LG",
    "LO",
    "MA",
    "MM",
    "MS",
    "NA",
    "NE",
    "NI",
    "OH",
    "OS",
    "PF",
    "PL",
    "RO",
    "SC",
    "SD",
    "SE",
    "SI",
    "SY"
]