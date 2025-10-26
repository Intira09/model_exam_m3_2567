# pip install pythainlp
# load > "/content/sample_data/example_dialect (3)(in) (1).csv" 

import pandas as pd
import requests
import time
import re
from sentence_transformers import SentenceTransformer, util
from pythainlp.tokenize import word_tokenize

# -----------------------------
# โหลด SBERT
# -----------------------------
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

articles = [
    """  สื่อสังคม (Social Media) หรือที่คนทั่วไปเรียกว่า สื่อออนไลน์ หรือ สื่อสังคม ออนไลน์ นั้น เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ ได้อย่างรวดเร็วไปยังผู้คนที่อยู่ทั่วทุกมุมโลกที่สัญญาณโทรศัพท์เข้าถึง เช่น การนําเสนอข้อดีนานาประการของสินค้าชั้นนํา สินค้าพื้นเมืองให้เข้าถึงผู้ซื้อได้
ทั่วโลก การนําเสนอข้อเท็จจริงของข่าวสารอย่างตรงไปตรงมา การเผยแพร่ งานเขียนคุณภาพบนโลกออนไลน์แทนการเข้าสํานักพิมพ์ เป็นต้น จึงกล่าวได้ว่า เราสามารถใช้สื่อสังคมออนไลน์ค้นหาและรับข้อมูลข่าวสารที่มีประโยชน์ได้เป็นอย่างดี
""",
    """อย่างไรก็ตาม หากใช้สื่อสังคมออนไลน์อย่างไม่ระมัดระวัง หรือขาดความรับผืดชอบต่อสังคมส่วนรวม ไม่ว่าจะเป็นการเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย การนำเสนอผลงานที่มีเนื้อหาล่อแหลมหรือชักจูงผู้รับสารไปในทางไม่เหมาะสม หรือการสร้างกลุ่มเฉพาะที่ขัดต่อศีลธรรมอันดีของสังคมตลอดจนใช้เป็นช่องทางในการกระทำผิดกฎหมายทั้งการพนัน การขายของ
ผิดกฎหมาย เป็นต้น การใช้สื่อสังคมออนไลน์ในลักษณะดังกล่าวจึงเป็นการใช้ที่เป็นโทษแก่สังคม
""",
    """ปัจจุบันผู้คนจํานวนไม่น้อยนิยมใช้สื่อสังคมออนไลน์เป็นช่องทางในการทํา การตลาดทั้งในทางธุรกิจ สังคม และการเมือง จนได้ผลดีแบบก้าวกระโดด ทั้งนี้ เพราะสามารถเข้าถึงกลุ่มคนทุกเพศ ทุกวัย และทุกสาขาอาชีพโดยไม่มีข้อจํากัดเรื่อง เวลาและสถานที่ กลุ่มต่างๆ ดังกล่าวจึงหันมาใช้สื่อสังคมออนไลน์เพื่อสร้างกระแสให้ เกิดความนิยมชมชอบในกิจการของตน ด้วยการโฆษณาชวนเชื่อทุกรูปแบบจนลูกค้า เกิดความหลงใหลข้อมูลข่าวสาร จนตกเป็นเหยื่ออย่างไม่รู้ตัว เราจึงควรแก้ปัญหา การตกเป็นเหยื่อทางการตลาดของกลุ่มมิจฉาชีพด้วยการเร่งสร้างภูมิคุ้มกันรู้ทันสื่อไม่ตกเป็นเหยื่อทางการตลาดโดยเร็ว
""",
    """แม้ว่าจะมีการใช้สื่อสังคมออนไลน์ในทางสร้างสรรค์สิ่งที่ดีให้แก่สังคม ตัวอย่างเช่น การเตือนภัยให้แก่คนในสังคมได้อย่างรวดเร็ว การส่งต่อข้อมูลข่าวสาร เพื่อระดมความช่วยเหลือให้แก่ผู้ที่กําลังเดือดร้อน เป็นต้น แต่หลายครั้งคนในสังคมก็ อาจรู้สึกไม่มั่นใจเมื่อพบว่าตนเองถูกหลอกลวงจากคนบางกลุ่มที่ใช้สื่อสังคมออนไลน์
เป็นพื้นที่แสวงหาผลประโยชน์ส่วนตัว จนทําให้เกิดความเข้าใจผิดและสร้างความ เสื่อมเสียให้แก่ผู้อื่น ดังนั้นการใช้สื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงจึงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสารโดยตรง
"""
]

# -----------------------------
# ใจความสำคัญ
# -----------------------------
main_idea_keywords = [
    "สื่อสังคมออนไลน์เป็นช่องทางที่ใช้ในการเผยแพร่ ค้นหา รับข้อมูลข่าวสาร",
    "การใช้สื่อสังคมออนไลน์อย่างไม่ระมัดระวังหรือขาดความรับผิดชอบจะเกิดโทษ ผลเสีย ข้อเสีย ผลกระทบหรือสิ่งไม่ดี",
    "ผู้ใช้ต้องรู้เท่าทันสื่อสังคมออนไลน์",
    "การใช้สื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสาร"
]

# -----------------------------
# โหลด local_word จาก CSV/Excel
# -----------------------------
def load_local_words(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    if "local_word" not in df.columns:
        raise ValueError("ไม่พบคอลัมน์ 'local_word' ในไฟล์")
    return [str(x).strip() for x in df["local_word"].dropna().tolist()]

# -----------------------------
# ฟังก์ชัน normalize ข้อความ
# -----------------------------
def normalize_text(words):
    text = str(words).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -----------------------------
# ฟังก์ชันตรวจการยกตัวอย่าง
# -----------------------------
def check_has_example(student_answer, local_words):
    student_text = normalize_text(student_answer)
    for w in local_words:
        w_norm = normalize_text(w)
        if w_norm in student_text:
            return True, w_norm
    return False, None


def detect_opinion(student_answer):
    """ตรวจคำบอกข้อคิดเห็น: 'ไม่เห็นด้วย' ก่อน 'เห็นด้วย'"""
    text = normalize_text(student_answer)
    if "ไม่เห็นด้วย" in text:
        return "ไม่เห็นด้วย", 1
    elif "เห็นด้วย" in text:
        return "เห็นด้วย", 1
    else:
        return "ไม่มีคำบอกข้อคิดเห็น", 0

# -----------------------------
# ฟังก์ชันประเมินคำตอบนักเรียน
# -----------------------------
def evaluate_student_answer(student_answer, articles, main_ideas, local_words):
    # แปลงเป็น str เผื่อ NaN หรือ float
    student_answer = str(student_answer)

    # ตรวจคำบอกข้อคิดเห็น
    opinion, score_opinion = detect_opinion(student_answer)

    # -------------------------
    # embedding SBERT
    # -------------------------
    article_embeddings = model.encode(articles, convert_to_tensor=True)
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    main_idea_embeddings = model.encode(main_ideas, convert_to_tensor=True)

    cosine_scores_articles = util.cos_sim(student_embedding, article_embeddings)
    cosine_scores_main_ideas = util.cos_sim(student_embedding, main_idea_embeddings)

    # -------------------------
    # ตรวจคัดลอกบทความ
    # -------------------------
    similarity_threshold_article = 0.87 #0.87
    copied_article = any(score.item() >= similarity_threshold_article for score in cosine_scores_articles[0])

    # -------------------------
    # ตรวจใจความสำคัญ
    # -------------------------
    similarity_threshold_main_idea = 0.55
    covered_main_idea_flags = [score.item() >= similarity_threshold_main_idea for score in cosine_scores_main_ideas[0]]
    has_main_idea = any(covered_main_idea_flags)

    # -------------------------
    # ตรวจการยกตัวอย่างจาก local_word
    # -------------------------
    has_example, found_example_word = check_has_example(student_answer, local_words)

    # -------------------------
    # ใช้ SSense ตรวจ sentiment
    # -------------------------
    url_ssense = "https://api.aiforthai.in.th/ssense"
    params = {"text": student_answer}
    headers_ssense = {"Apikey": "zyHC3BNtLiesIuTj2UMlQd8DhrVXBxzM"}
    try:
        response_sentiment = requests.get(url_ssense, headers=headers_ssense, params=params, timeout=10)
        sentiment_result = response_sentiment.json()
    except:
        sentiment_result = {"polarity-pos": False, "polarity-neg": True, "score": 50}

    # -------------------------
    # นับจำนวนบรรทัด
    # -------------------------
    line_count = student_answer.count("\n") + 1

    # -------------------------
    # ตรวจ keyword ให้คะแนน 2
    # -------------------------
    keyword_words_2 = ["เพราะ"]
    has_keyword_2 = any(k in student_answer for k in keyword_words_2)

    # ✅ ดึงคำหลัง "เพราะ"
    after_keyword_2 = "0"
    match = re.search(r"เพราะ\s*(.*)", student_answer)
    if match:
        # เอาส่วนที่เหลือหลัง "เพราะ"
        text_after = match.group(1).strip()
        # แยกเป็นคำ
        words = text_after.split()
        # เอา 10 คำแรก (ถ้าน้อยกว่า 10 คำก็เอาทั้งหมด)
        after_keyword_2 = " ".join(words[:10]) if words else "0"

    keyword_words_0 = ["เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ", "ตกเป็นเหยื่อล่อ"]
    has_keyword_0 = any(k in student_answer for k in keyword_words_0)

    keyword_words_2_donthave = ["จำเป็น", "สิ่งที่ดี", "ข้อดี", "ป้องกัน"]
    has_keyword_2_donthave = any(k in student_answer for k in keyword_words_2_donthave)

    keyword_words_2_donthave_de = ["โกง", "ฟ้อง", "ไม่น่าเชื่อถือ", "ป้องกัน", "เครื่องมือ",
                                  "โดนชักจูง", "ถูกกระทำ", "เป็นภัย", "มีความผิด",
                                  "ทางที่เหมาะสม","หลอกเอาเงิน", "คนที่ทำไม่ดี",
                                  "ของปลอม", "ไม่มีความรับผิดชอบ", "การเตือนเรื่องภัย"]
    has_keyword_2_donthave_de = any(k in student_answer for k in keyword_words_2_donthave_de)

    # เพิ่มคีย์เวิร์ดคำ ให้คะแนน 4 ถ้าพบ
    # -------------------------
    keyword_words_4 = ["เจตนาแอบแฝง", "เจตนา", "แอบแฝง", "มิจฉาชีพ", "ผ่อนคลาย",
                      "ผิดกฎหมาย",  "พนัน", "หลอก", "จำคุก", "โฆษณาชวนเชื่อ",
                      "ถูกหมายเรียก", "ติดโรค", "เจาะข้อมูล"]
    has_keyword_4 = any(k in student_answer for k in keyword_words_4)

    # เพิ่มคีย์เวิร์ดคำ ให้คะแนน 8 ถ้าพบ
    # -------------------------
    keyword_words_6 = ["การใช้สื่อสังคมออนไลน์ในทางที่ผิด", "โดนมิจฉาชีพหลอก", "ตกเป็นเหยื่อของสังคมออนไลน์",
                       "การพนันหรือขายของผิดกฏหมาย", "การตลาดทั้งในธุรกิจ สังคม และการเมือง",
                       "สินค้าพื้นเมือง", "การโฆษณาสินค้า", "ขัดต่อศีลธรรมอันดีของสังคม",
                       "การเตือนภัยให้แก่คนในสังคม" , "เขียนวิจารณ์", "รู้เท่าทันสื่อออนไลน์",
                       "กลลวงมิจฉาชีพ", "วิจารณ์ในทางที่เสียหาย", "รับชมข่าวสาร",
                       "ประกอบอาชีพ", "การส่งข้อความ"]
    has_keyword_6 = any(k in student_answer for k in keyword_words_6)

    # เพิ่มคีย์เวิร์ดคำ ให้คะแนน 8 ถ้าพบ
    # -------------------------
    keyword_words_8 = ["ไลฟ์สด" , "ดูหนัง", "การศึกษา", "เปิดเพลงฟัง", "แพลตฟอร์มออนไลน์",
                      "ไลฟ์", "เพลง", "โรงงาน", "ท่องเที่ยว", "การสั่งของจากสื่อ", "โอนเงิน", "พัฒนาตนเอง"]
    has_keyword_8 = any(k in student_answer for k in keyword_words_8)

    # -------------------------
    # ให้คะแนนตามเงื่อนไข
    # -------------------------
    polarity_pos = sentiment_result.get("polarity-pos", False)
    polarity_neg = sentiment_result.get("polarity-neg", True)

    if sentiment_result.get("polarity", "") == "positive" or (polarity_pos and not polarity_neg):
        sentiment = "pos"
    else:
        sentiment = "neg"

    # ✅ ตั้งค่าเริ่มต้น ป้องกัน UnboundLocalError
    score_total = 0

    if (
        (opinion == "ไม่มีคำบอกข้อคิดเห็น" and has_keyword_2 and
         after_keyword_2 == "0" and not has_main_idea
        ) or
        (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not has_keyword_2 and
         after_keyword_2 == "0"  and has_main_idea
        ) or
        (opinion == "ไม่มีคำบอกข้อคิดเห็น" and has_keyword_2 and
         after_keyword_2 == "0" and not has_main_idea and not has_example
        ) or
        (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"]  and not has_main_idea and
         not has_keyword_4 and not has_example and  found_example_word is None
         and not has_keyword_8 and not has_keyword_6
         and not has_keyword_2_donthave_de and not has_keyword_2_donthave
        ) or
        (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and after_keyword_2 == "0" and
         not has_keyword_4 and not has_example and  found_example_word is None and
         not has_keyword_8 and not has_keyword_6
         and not has_keyword_2_donthave_de and not has_keyword_2_donthave
        ) or (has_keyword_0)
    ):
        score_total = 0

    elif (
        (
            (opinion == "ไม่เห็นด้วย" and has_keyword_2_donthave)
            or (opinion == "เห็นด้วย" and has_keyword_2_donthave_de)
        )
        and has_main_idea
        and sentiment == "neg"
        and not copied_article
    ):
        score_total = 2

    elif (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"]
          and has_example and not has_keyword_6
          and has_keyword_2 and has_keyword_8):
            score_total = 8

    elif (
          opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and
           (copied_article or has_example) and
          has_main_idea and not has_keyword_8 and has_keyword_6
    ):
        score_total = 6

    elif (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"]
          and (not found_example_word or not has_example or has_keyword_4)):
        score_total = 4

    elif (
          (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not copied_article
          and sentiment == "neg" and has_main_idea) or
          (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not copied_article
          and sentiment == "neg"and has_main_idea and not has_keyword_2)
    ):
        score_total = 2

    elif (
            (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and not copied_article
            and not has_main_idea and not found_example_word
             and not has_example)
    ):
        score_total = 2

    elif (
        (opinion == "ไม่มีคำบอกข้อคิดเห็น" and copied_article)
        or (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not has_main_idea)
        or (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and not has_main_idea)
    ):
        score_total = 0


    return {
        "opinion": opinion,
        "score_opinion": score_opinion,
        "has_example": has_example,
        "found_example_word": found_example_word,
        "sentiment": sentiment_result,
        "copied_article": copied_article,
        "has_main_idea": has_main_idea,
        "has_keyword_2": has_keyword_2,
        "after_keyword_2": after_keyword_2,
        "has_keyword_2_donthave": has_keyword_2_donthave,
        "has_keyword_2_donthave_de": has_keyword_2_donthave_de,
        "has_keyword_4": has_keyword_4,
        "has_keyword_6": has_keyword_6,
        "has_keyword_8": has_keyword_8,
        "line_count": line_count,
        "score_total": score_total
    }

student_answer = """     เห็นด้วย   เพราะมันผิดกฎหมายถ้าไม่ระวังให้ดี เราอาจจะเสียหายเป็นอย่างมาก
การวิพากษ์วิจารณ์ ผู้อื่นทำให้เกิดความเสียหายหนักมาก ผู้ที่ถูกวิจารณ์ จะไม่มีหน้าไปพบ
ปะผู้อื่น ไม่ควรชักจูงผู้อื่น ให้มาทำในสิ่งที่ไม่ดี และผิดกฎหมายทำให้ผู้อื่นเดือดเปินอย่าง
มาก การสร้างกระแส ที่ไม่ดีและมันอาจจะส่งผลกระทบผู้อื่นได้ เราไม่ควรเชื่ลฟังสิ่งที่ไม่ดี
ถ้าเป็นไปได้หลีกเลี่ยงจะดีกว่า เราไม่ควรส่งต่อสื่ลที่ไม่ดีสื่อออนไลน์จะชอบมีเบอร์แปลกโทร
เข้ามาทางที่ดีไม่รับจะดีกว่า อาจจะตกเป็นเหยื่อล่อก็เป็นไปได้ ดังนั้นเราควรรักษา
สิทธิของตนเองให้ดี เพราะสื่อออนไลน์มีหลายประเภทที่เราไม่รู้ละเข้ามาแบบไหน"""

# -----------------------------
# โหลดคำยกตัวอย่าง
# -----------------------------
local_words = load_local_words("/content/sample_data/example_dialect (3)(in) (1).csv")

result = evaluate_student_answer(student_answer, articles, main_idea_keywords, local_words)

print("ผลลัพธ์การประเมิน:")
for k, v in result.items():
    print(f"{k}: {v}")
