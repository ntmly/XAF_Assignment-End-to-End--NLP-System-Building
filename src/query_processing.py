import re
from typing import Dict

ACRONYM_DICT: Dict[str, str] = {

    # ----- Tên trường -----
    r'\bđhqghn\b':           'Đại học Quốc gia Hà Nội',
    r'\bđhqg\b':             'Đại học Quốc gia Hà Nội',
    r'\bvnu\b':              'Đại học Quốc gia Hà Nội',
    r'\buet\b':              'Trường Đại học Công nghệ',
    r'\bđhcn\b':             'Trường Đại học Công nghệ',
    r'\bđh công nghệ\b':     'Trường Đại học Công nghệ',
    r'\bđhnn\b':             'Trường Đại học Ngoại ngữ',
    r'\bđhkhxhnv\b':         'Trường Đại học Khoa học Xã hội và Nhân văn',
    r'\bđhkhtn\b':           'Trường Đại học Khoa học Tự nhiên',
    r'\bđhkt\b':             'Trường Đại học Kinh tế',
    r'\bđhgd\b':             'Trường Đại học Giáo dục',

    # ----- Tuyển sinh / xét tuyển -----
    r'\bhsa\b':              'đánh giá năng lực',
    r'\bđgnl\b':             'đánh giá năng lực',
    r'\bthpt\b':             'trung học phổ thông',
    r'\bsat\b':              'SAT',
    r'\bielts\b':            'IELTS',
    r'\balevel\b':           'A-level',
    r'\bact\b':              'ACT',

    # ----- Ngành học (mã tuyển sinh) -----
    r'\bcntt\b':             'công nghệ thông tin',
    r'\bktmt\b':             'kỹ thuật máy tính',
    r'\bttnt\b':             'trí tuệ nhân tạo',
    r'\bkhmt\b':             'khoa học máy tính',
    r'\bkhdl\b':             'khoa học và kỹ thuật dữ liệu',
    r'\bktvl\b':             'vật lý kỹ thuật',
    r'\bktck\b':             'cơ kỹ thuật',
    r'\bktdt\b':             'kỹ thuật điện tử viễn thông',
    r'\bktnl\b':             'kỹ thuật năng lượng',
    r'\bhttt\b':             'hệ thống thông tin',
    r'\bmmt\b':              'mạng máy tính và truyền thông dữ liệu',
    r'\bktr\b':              'kỹ thuật robot',
    r'\bktdh\b':             'kỹ thuật điều khiển và tự động hóa',
    r'\bcnnn\b':             'công nghệ nông nghiệp',
    r'\bcnsh\b':             'công nghệ kỹ thuật sinh học',
    r'\bcnhk\b':             'công nghệ hàng không vũ trụ',
    r'\btkdh\b':             'thiết kế công nghiệp và đồ họa',
    r'\bcnvl\b':             'công nghệ vật liệu và vi điện tử',
    r'\bcnxd\b':             'công nghệ kỹ thuật xây dựng',
    r'\bktem\b':             'công nghệ kỹ thuật cơ điện tử',

    # ----- Học vụ / quy chế -----
    r'\btc\b':               'tín chỉ',
    r'\bgpa\b':              'điểm trung bình tích lũy',
    r'\bđtb\b':              'điểm trung bình',
    r'\bđtbtl\b':            'điểm trung bình tích lũy',
    r'\bcđr\b':              'chuẩn đầu ra',
    r'\bkltn\b':             'khóa luận tốt nghiệp',
    r'\bđatn\b':             'đồ án tốt nghiệp',
    r'\bpđt\b':              'phòng đào tạo',
    r'\bcvht\b':             'cố vấn học tập',
    r'\bktx\b':              'ký túc xá',
}

# ============================================================
# TỪ ĐỒNG NGHĨA 
# ============================================================
SYNONYM_DICT: Dict[str, str] = {

    # Lịch sử
    'thành lập':        'thành lập founded established năm thành lập',
    'tiền thân':        'tiền thân lịch sử hình thành nguồn gốc',
    'lịch sử':          'lịch sử quá trình hình thành phát triển',
    'mã đại học': 'mã đại học mã trường QH mã cơ sở đào tạo',

    # Tuyển sinh
    'điểm chuẩn':       'điểm chuẩn điểm trúng tuyển điểm xét tuyển',
    'xét tuyển':        'xét tuyển tuyển sinh phương thức xét tuyển',
    'chỉ tiêu':         'chỉ tiêu số lượng tuyển sinh chỉ tiêu tuyển sinh',
    'điểm sàn':         'điểm sàn ngưỡng đầu vào điểm tối thiểu',
    'học bổng':         'học bổng hỗ trợ tài chính miễn giảm học phí',
    'ký túc xá':        'ký túc xá ktx chỗ ở sinh viên nội trú',
    'nhập học':         'nhập học xác nhận nhập học thủ tục nhập học',
    'ngày tháng năm':    'ngày tháng năm thành lập quyết định ký',
    'sư phạm ngoại ngữ': 'sư phạm ngoại ngữ đại học sư phạm ngoại ngữ hà nội 1967',
    'email':             'email địa chỉ liên hệ phòng đào tạo daotao',
    'phòng đào tạo':     'phòng đào tạo pđt daotao_dhcn vnu liên hệ',
    'mã cơ sở':          'mã cơ sở mã tuyển sinh qhi',

    # Học phí
    'học phí':          'học phí mức phí chi phí học tập đóng học phí',

    # Học vụ / quy chế
    'đăng ký học phần': 'đăng ký học phần đăng ký môn học',
    'học lại':          'học lại đăng ký học lại điểm f',
    'cải thiện điểm':   'cải thiện điểm học cải thiện điểm d',
    'tín chỉ':          'tín chỉ số tín chỉ khối lượng học tập',
    'điểm trung bình':  'điểm trung bình gpa điểm tích lũy',
    'xử lý học vụ':     'xử lý học vụ cảnh báo học vụ buộc thôi học',
    'tốt nghiệp':       'tốt nghiệp điều kiện tốt nghiệp xét tốt nghiệp',
    'khóa luận':        'khóa luận đồ án tốt nghiệp kltn đatn',
    'miễn học':         'miễn học miễn học phần điều kiện miễn',
    'ngoại ngữ':        'ngoại ngữ tiếng anh ielts chuẩn đầu ra ngoại ngữ',
    'chuyển trường':    'chuyển trường chuyển ngành chuyển chương trình',
    'cố vấn học tập':   'cố vấn học tập cvht hỗ trợ sinh viên',

    # Cơ sở vật chất
    'hòa lạc':          'hòa lạc cơ sở hòa lạc campus hòa lạc',
    'cơ sở':            'cơ sở campus địa điểm khuôn viên trụ sở',
    'địa chỉ':          'địa chỉ trụ sở vị trí cơ sở',
    'chủ nhiệm khoa': 'chủ nhiệm khoa trưởng khoa giảng viên lãnh đạo khoa',
    'công nghệ thông tin': 'công nghệ thông tin cntt khoa cntt',

    # Nghiên cứu khoa học
    'nghiên cứu':       'nghiên cứu khoa học nckh sinh viên nghiên cứu',
    'điểm thưởng':      'điểm thưởng thành tích nghiên cứu khoa học',
}


def preprocess_query(query: str) -> str:
    """
    Tiền xử lý câu hỏi:
    1. Chuẩn hóa khoảng trắng, bỏ ký tự thừa
    2. Mở rộng viết tắt tiếng Việt
    3. Mở rộng từ đồng nghĩa để cải thiện BM25 recall
    """
    # 1. Chuẩn hóa
    query = query.strip()
    query = re.sub(r'\s+', ' ', query)       
    query = re.sub(r'[?!。？！]+$', '', query).strip() 
    query_lower = query.lower()

    # 2. Mở rộng viết tắt (case-insensitive)
    for pattern, expansion in ACRONYM_DICT.items():
        query_lower = re.sub(pattern, expansion.lower(), query_lower)

    # 3. Mở rộng từ đồng nghĩa
    extra_terms = []
    for keyword, expansion in SYNONYM_DICT.items():
        if keyword in query_lower:
            extra_terms.append(expansion)

    if extra_terms:
        query_lower = query_lower + ' ' + ' '.join(extra_terms)

    return query_lower


def expand_query(query: str) -> str:
    """Alias để không cần sửa import cũ."""
    return preprocess_query(query)