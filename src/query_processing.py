import re
from typing import Dict

ACRONYM_DICT: Dict[str, str] = {
    r'\bvnu\b': 'Vietnam National University, Hanoi',
    r'\buet\b': 'University of Engineering and Technology',
    r'\bcmu\b': 'Carnegie Mellon University',
    r'\bpit\b': 'Pittsburgh',
    r'\bicml\b': 'International Conference on Machine Learning'
}

def preprocess_query(query: str) -> str:
    """Tiền xử lý câu hỏi: lower, mở rộng từ viết tắt"""
    query = query.lower().strip()
    for pattern, expansion in ACRONYM_DICT.items():
        query = re.sub(pattern, expansion, query)
    return query

def expand_query(query: str) -> str:
    """Mở rộng query bằng từ đồng nghĩa đơn giản"""
    expansions = {
        'thành lập': 'thành lập established founded',
        'trường': 'university school college',
        'năm': 'year date'
    }
    for key, val in expansions.items():
        if key in query:
            query += ' ' + val
    return query