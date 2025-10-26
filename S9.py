#pip install pythainlp

import re
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_words
from sentence_transformers import SentenceTransformer, util

# ------------------ ฟังก์ชันพื้นฐาน ------------------
def preprocess_text(text):
    return "".join(text.split())

def is_thai_word(word):
    return bool(re.fullmatch(r'[\u0E00-\u0E7F]+', word))

# ------------------ ตรวจคำ / กฎเฉพาะ ------------------
def check_thai_text_integrity(text, ignore_single_char=None):
    """
    ตรวจคำเดี่ยวและคืนคำเดี่ยวพร้อมคำนำหน้า
    :param text: ข้อความนักเรียน
    :param ignore_single_char: list ของคำที่เป็นคำนำหน้า ถ้าคำนำหน้าเป็นคำนี้ จะไม่ถือว่าผิด
    :return: single_char_words (list of dict: {'word': w, 'preceding': prev_word}), special_violations
    """
    if ignore_single_char is None:
        ignore_single_char = []

    text_clean = preprocess_text(text)
    words = word_tokenize(text_clean, engine='newmm')  # tokenized
    thai_words_only = [w for w in words if is_thai_word(w)]

    single_char_words = []
    for i, w in enumerate(thai_words_only):
        if len(w) == 1:
            preceding = thai_words_only[i-1] if i > 0 else None
            # ถ้าคำนำหน้าอยู่ใน ignore_single_char → ข้าม
            if preceding in ignore_single_char:
                continue
            single_char_words.append({"word": w, "preceding": preceding})

    special_violations = []

    return single_char_words, special_violations

# ------------------ ตรวจความสัมพันธ์แบบบรรทัด ------------------
paraphrase_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def semantic_similarity_lines(text, threshold=0.3):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    embeddings = paraphrase_model.encode(lines)

    failed_lines = []
    for i, emb in enumerate(embeddings):
        sims = [util.cos_sim(emb, embeddings[j]).item() for j in range(len(embeddings)) if j != i]
        max_sim = max(sims) if sims else 1.0
        if max_sim < threshold:
            failed_lines.append({
                "line_index": i,
                "line_text": lines[i],
                "max_similarity": round(max_sim, 3)
            })
    return failed_lines

# ------------------ ตรวจคำซ้ำ / n-grams ------------------
def ngrams(tokens, n):
    return ["".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def find_repeated_ngrams(tokens, min_len=2, ignore_list=None):
    """
    คืน dict ของ n-grams ซ้ำ พร้อมจำนวนครั้ง
    """
    if ignore_list is None:
        ignore_list = []
    repeated = {}
    max_n = len(tokens)
    for n in range(min_len, max_n+1):
        ngs = ngrams(tokens, n)
        counts = {}
        for ng in ngs:
            if any(ignore in ng for ignore in ignore_list):
                continue
            counts[ng] = counts.get(ng, 0) + 1
        for ng, c in counts.items():
            if c > 1:
                repeated[ng] = c
    # เก็บเฉพาะคำที่ยาวที่สุด
    longest_only = {}
    for ng, count in repeated.items():
        if not any((ng in other and len(other) > len(ng)) for other in repeated):
            longest_only[ng] = count
    # แปลงเป็นจำนวนคำที่ซ้ำ (นับเป็นคำละ 1 คะแนน)
    repeated_words_count = sum(count - 1 for count in longest_only.values())
    return {"repeated_ngrams": longest_only, "count": repeated_words_count}

def find_specific_terms(text, specific_terms):
    """
    คืน dict ของคำเฉพาะที่พบ พร้อมจำนวนครั้งที่เกิน 1 และจำนวนคำที่เจอ
    """
    found = {}
    total_count = 0
    for term in specific_terms:
        count = text.count(term)
        if count > 1:
            found[term] = count
            total_count += count - 1  # นับคำที่เกิน 1
    return {"specific_found": found, "count": total_count}


# ------------------ ฟังก์ชันรวมตรวจคำตอบนักเรียน 1 คน ------------------
def evaluate_student_answer(student_text, ignore_list=None, specific_terms=None, ignore_single_char=None, similarity_threshold=0.3):
    if ignore_list is None:
        ignore_list = []
    if specific_terms is None:
        specific_terms = []

    # ---- ตรวจคำเดี่ยว, unknown, special rules ----
    single_char_words, special_violations = check_thai_text_integrity(student_text, ignore_single_char)
    missing_content = {
        "single_char_words": single_char_words,
        "special_violations": special_violations
    }

    # ---- ตรวจคำซ้ำ / n-grams ----
    s_clean = student_text.replace("\n", "")
    tokens = [t for t in word_tokenize(s_clean, keep_whitespace=False) if t.strip()]
    repeated_ngrams_result = find_repeated_ngrams(tokens, min_len=2, ignore_list=ignore_list)
    specific_found_result = find_specific_terms(s_clean, specific_terms)

    duplicate_content = {
        "repeated_ngrams": repeated_ngrams_result["repeated_ngrams"],
        "specific_found": specific_found_result["specific_found"]
    }

    # ---- ตรวจความสัมพันธ์แบบบรรทัด ----
    failed_similarity = semantic_similarity_lines(student_text, threshold=similarity_threshold)
    semantic_issue = failed_similarity  # JSON แสดงเฉพาะบรรทัดที่ไม่ผ่าน

    # ---- คำนวณคะแนน ----
    # ---- คำนวณคะแนน ----
    score = 3
    # หักคะแนนทีละ 1 ต่อปัญหา แต่สำหรับ repeated/specific นับคำที่เจอ
    if single_char_words:
        score -= 1
    if special_violations:
        score -= 1
    score -= repeated_ngrams_result["count"]
    score -= specific_found_result["count"]
    if failed_similarity:
        score -= 1
    score = max(0, score)

    return {
        "เนื้อความขาด": missing_content,
        "เนื้อความซ้ำ": duplicate_content,
        "เนื้อความไม่สัมพันธ์กัน": semantic_issue,
        "คะแนนรวม": score
    }

# ------------------ ตัวอย่างการใช้งาน ------------------
ignore_list = ["สื่อ", "สื่อออนไลน์", "สื่อสังคม", "สื่อสังคมออนไลน์",
               "ออนไลน์", "ออนไลท์", "\n", "ในบ้าน", "ในรูปแบบ",
               "การใช้", "เด็กชายA", "ที่ดี", "ให้แก่", "หากเรา",
               "ได้อย่าง", "ในการ", "เราก็", "อย่างรวดเร็ว", "ในทาง",
               "ได้ด้วย", "ก็มี", "ที่ไม่", "หรือจะ", "ในปัจจุบัน", "สามารถดู",
               "ก็ไม่", "เช่นการ", "สามารถใช้", "ใช้เพื่อ", "ควรใช้",
               "ของตน", "ที่", "ไม่ควร", "จึง", "ควร", "เรา", "ใช้อย่าง", "ชาชีพ",
               "นั้นสามารถ", "อีกมากมาย", "นั้นใช้", "การนำเสนอ", "ก็จะ", "ถ้าใช้",
               "อยากจะ", "ก็ทำได้", "สังคมเป็น", "ช่วยให้", "เข้าถึงได้", "หรือ",
               "อย่างมาก", "ในทุกวันนี้", "ต่างๆ", "เพราะ", "อาจ", "เข้ามาช่วยทำงานในด้าน" ,
               "ช่วย","ใน", "ตรงไปตรงมา", "เกิดประโยชน์", "แก่สังคม", "จะทำให้", "สือ",
               "มีการ", "ไม่รู้","ความ", "ไม่น้อย", "คือ", "ไหน", "แก่", "ตนเอง",
               "ซื้อของ", "ตรวจสอบว่า","ได้ง่าย", "ค้นคว้า", "ข้อมูล", "กลุ่มแชท",
               "การเล่น", "ด้วยเช่นกัน", "ให้กับ", "และ", "มีทั้ง", "เช่น", "คน",
               "ปรโยชน์", "อะไรได้", "สมัยนี้มี", "สังคมส่วนรวม", "ตามโฆษณา", "ไม่มี",
               "ให้ดี", "ทำอะไร", "เค้า", "ซี", "แอป", "ผิดการ",
               "มีด้าน", "จะได้", "ได้ไม่ต้อง", "โดยไม่", "เป็นสิ่ง", "ไม่ดี",
               "คุยกับเพื่อน", "บางกลุ่ม", "โทษต่อ", "อย่างระมัดระวัง",
               "คุย", "เพื่อน", "โพช", "เช็กให้"]

specific_terms = []

ignore_single_char = ["สิ", "สี่", "สัญญา", "ผิดก", "หริ", "รู", "ภูมิ",
                      "เจ", "คา", "เป้", "เสีย", "หาย", "ผิด", "ที",
                      "สี" , "ริ" , "ข่อ" , "ออนไลน์", "โท", "ต่อ", "ใส",
                      "ข่า", "แอ", "ยุค", "หลาย", "ฮิต", "ทู",
                      "บุ", "ยี่", "มาก", "ทำได้", "ตน", "เขา",
                      "ควร", "ดัก", "กรู", "กลุ่ม", "เด็ก", "โคล", "มั่ว", "คน",
                      "อย่า", "รู้", "รี", "โพ", "เฉย", "เยอ", "ดี", "ติ",
                      "ลื่อ", "ดี", "เทคโนโลยี", "กุ", "ผู้อื่น", "เสียหาย",
                      "สิ้น", "ค้น", "คอ", "หลอ"]  # คำเดี่ยวที่ไม่ถือว่าผิด

# , ""
student_text = """เห็นด้วยเพราะปัจจุบันสื่อออนไลน์มีการพัฒนามากขึ้นเรื่อยๆทั้งการทำงานหรือหลายๆอย่างแต่การเล่นสื่อออนไลน์หรือ
แอปหลายๆอย่างในปัจจุบันควรระมัดระวังให้มากขึ้นเพราะเดียวนี้สื่อออนไลน์สามารถทำได้หลายอย่างมากทั้งหลอกเงิน
หรือโกงเงินควรเช็ก ให้ดีก่อนจะทำอะไรหลายๆอย่างควรระมัดระวังมากขึ้นปัจจุบันผู้คนส่วนมากมักใช้ในการทำงานหรือ
หาขว้มูลสะส่วนใหญ่การทำงานในสื่อออนไลน์สามารถทำให้จานก้าวกระโดดได้ดีและไวขึ้นหรือการเล่นแวปต่างๆก็สาวารกเช้า
ถึงได้ทุกเพศทุกวัยแต่ควรเล่นให้เป็นเวลาไม่ติดจนเกินไปและระมัดระวังในการเล่นแอปต่างๆควรมีความรับผิดชอบ
ไม่ทำในสิ่งที่ผิดกฎหมายหรือชักชวนผู้อื่นในทางที่ผิดการโฆษณาควรโฆษณาในควรมาเป็นจริงไม่โกหกหรือหลอก
ลวงเพราะมิน่า หลายคนที่ไม่รู้และถูกชักจูงเข้ามาในทางที่ผิดการขอ้ความหรือข้อมูลต่างๆก็ควรเช็กให้ดีก่อนที่
จะทำในสิ่งนั้นๆ ควรถูกข้อมูลให้ดีก่อนจะทำอะไรหลายๆ อย่าง"""

result = evaluate_student_answer(
    student_text,
    ignore_list=ignore_list,
    specific_terms=specific_terms,
    ignore_single_char=ignore_single_char,
    similarity_threshold=0.3
)

print(result)
