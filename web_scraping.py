import arxiv
from datetime import datetime, timedelta, timezone

def fetch_new_papers(query="cat:cs.AR OR cat:cs.LG", max_results=50, days_back=7):
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    papers = []
    for result in search.results():
        if result.published >= start_date:
            papers.append({
                'title': result.title,
                'abstract': result.summary,
                'link': result.entry_id,
                'pdf': result.pdf_url,
                'published': result.published
            })
    return papers