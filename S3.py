# pip install pythainlp

import re
import pandas as pd
import requests
from pythainlp.tokenize import word_tokenize
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- ตั้งค่า ----------
API_KEY = '' # add api
TNER_URL = 'https://api.aiforthai.in.th/tner'

# ---------- โหลด Dataset ----------
examples_df = pd.read_csv('/content/sample_data/example_dialect (3).csv')  # 'local_word'
pronouns_df = pd.read_csv('/content/sample_data/personal_pronoun (1).csv')  # 'personal pronoun 1', 'personal pronoun 2'

example_phrases = examples_df['local_word'].dropna().tolist()
pronouns_1 = pronouns_df['personal pronoun 1'].dropna().tolist()
pronouns_2 = pronouns_df['personal pronoun 2'].dropna().tolist()
pronouns_1_2 = pronouns_1 + pronouns_2

# ---------- บทความอ้างอิง ----------
reference_text = """
สื่อสังคม (Social Media) หรือที่คนทั่วไปเรียกว่า สื่อออนไลน์ หรือ สื่อสังคม ออนไลน์ นั้น เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ ได้อย่างรวดเร็วไปยังผู้คนที่อยู่ทั่วทุกมุมโลกที่สัญญาณโทรศัพท์เข้าถึง เช่น การนําเสนอข้อดีนานาประการของสินค้าชั้นนํา สินค้าพื้นเมืองให้เข้าถึงผู้ซื้อได้
ทั่วโลก การนําเสนอข้อเท็จจริงของข่าวสารอย่างตรงไปตรงมา การเผยแพร่ งานเขียนคุณภาพบนโลกออนไลน์แทนการเข้าสํานักพิมพ์ เป็นต้น จึงกล่าวได้ว่า เราสามารถใช้สื่อสังคมออนไลน์ค้นหาและรับข้อมูลข่าวสารที่มีประโยชน์ได้เป็นอย่างดี
  อย่างไรก็ตาม หากใช้สื่อสังคมออนไลน์อย่างไม่ระมัดระวัง หรือขาดความรับผืดชอบต่อสังคมส่วนรวม ไม่ว่าจะเป็นการเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย การนำเสนอผลงานที่มีเนื้อหาล่อแหลมหรือชักจูงผู้รับสารไปในทางไม่เหมาะสม หรือการสร้างกลุ่มเฉพาะที่ขัดต่อศีลธรรมอันดีของสังคมตลอดจนใช้เป็นช่องทางในการกระทำผิดกฎหมายทั้งการพนัน การขายของ
ผิดกฎหมาย เป็นต้น การใช้สื่อสังคมออนไลน์ในลักษณะดังกล่าวจึงเป็นการใช้ที่เป็นโทษแก่สังคม
	ปัจจุบันผู้คนจํานวนไม่น้อยนิยมใช้สื่อสังคมออนไลน์เป็นช่องทางในการทํา การตลาดทั้งในทางธุรกิจ สังคม และการเมือง จนได้ผลดีแบบก้าวกระโดด ทั้งนี้ เพราะสามารถเข้าถึงกลุ่มคนทุกเพศ ทุกวัย และทุกสาขาอาชีพโดยไม่มีข้อจํากัดเรื่อง เวลาและสถานที่ กลุ่มต่างๆ ดังกล่าวจึงหันมาใช้สื่อสังคมออนไลน์เพื่อสร้างกระแสให้ เกิดความนิยมชมชอบในกิจการของตน ด้วยการโฆษณาชวนเชื่อทุกรูปแบบจนลูกค้า เกิดความหลงใหลข้อมูลข่าวสาร จนตกเป็นเหยื่ออย่างไม่รู้ตัว เราจึงควรแก้ปัญหา การตกเป็นเหยื่อทางการตลาดของกลุ่มมิจฉาชีพด้วยการเร่งสร้างภูมิคุ้มกันรู้ทันสื่อไม่ตกเป็นเหยื่อทางการตลาดโดยเร็ว
	แม้ว่าจะมีการใช้สื่อสังคมออนไลน์ในทางสร้างสรรค์สิ่งที่ดีให้แก่สังคม ตัวอย่างเช่น การเตือนภัยให้แก่คนในสังคมได้อย่างรวดเร็ว การส่งต่อข้อมูลข่าวสาร เพื่อระดมความช่วยเหลือให้แก่ผู้ที่กําลังเดือดร้อน เป็นต้น แต่หลายครั้งคนในสังคมก็ อาจรู้สึกไม่มั่นใจเมื่อพบว่าตนเองถูกหลอกลวงจากคนบางกลุ่มที่ใช้สื่อสังคมออนไลน์
เป็นพื้นที่แสวงหาผลประโยชน์ส่วนตัว จนทําให้เกิดความเข้าใจผิดและสร้างความ เสื่อมเสียให้แก่ผู้อื่น ดังนั้นการใช้สื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงจึงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสารโดยตรง
"""

# ---------- โหลดโมเดล WangchanBERTa ----------
model = SentenceTransformer("airesearch/wangchanberta-base-att-spm-uncased")

# ---------- ฟังก์ชันตรวจแต่ละเงื่อนไข ----------
def call_tner(text):
    headers = {'Apikey': API_KEY}
    data = {'text': text}
    try:
        resp = requests.post(TNER_URL, headers=headers, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"TNER API error: {e}")
    return None

def check_summary_similarity(student_answer, reference_text, threshold=0.8):
    # ใช้ SBERT วัด cosine similarity
    embeddings = model.encode([student_answer, reference_text])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold

def check_examples(student_answer, example_phrases):
    return not any(phrase in student_answer for phrase in example_phrases)

def check_pronouns(student_answer, pronouns_list):
    words = word_tokenize(student_answer, engine='newmm')
    return not any(p in words for p in pronouns_list)

def check_abbreviations(student_answer):
    pattern = r'\b(?:[ก-ฮA-Za-z]\.){2,}'
    if re.search(pattern, student_answer):
        return False
    tner_result = call_tner(student_answer)
    if tner_result:
        for item in tner_result.get('entities', []):
            if item['type'] in ['ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB']:
                return False
    return True

def check_title(student_answer, forbidden_title="การใช้สื่อสังคมออนไลน์"):
    return forbidden_title not in student_answer

def validate_student_answer(student_answer):
    results = {
        "summary_similarity": check_summary_similarity(student_answer, reference_text),
        "no_example": check_examples(student_answer, example_phrases),
        "no_pronouns": check_pronouns(student_answer, pronouns_1_2),
        "no_abbreviations": check_abbreviations(student_answer),
        "no_title": check_title(student_answer),
    }
    errors = [k for k, v in results.items() if not v]
    score = 1 if len(errors) == 0 else 0
    return score, ', '.join(errors) if errors else 'ผ่านทุกเงื่อนไข'

# ตัวอย่างการใช้
student_answer = """
สื่อสังคมหรือมีชื่อเรียกว่า (Social Media) หรือที่คนอื่นเรียกว่า สื่อออนไลน์
หรือสื่อสังคมออนไลน์เป็นสื่อที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆได้อย่างรวดเร็ว
ผู้คนที่อยู่ทั่วทุกมุมโลกที่สัญญาณโทรศัพท์เข้าถึง เช่น นำเสนอนานาประการของสิ้นค้าชั้น-
นำ ให้เข้าถึงผู้ซื้อได้ทั่วโลก จึงกล่าวได้ว่าสามารถใช้สื่อสังคมออนไลน์ค้นหาข้อมูลข่าวสาร
ที่มีประโยชน์ได้เป็นอย่างดี
"""

score, error_conditions = validate_student_answer(student_answer)
print(f"คะแนน: {score}")
if score == 0:
    print("ผิดเงื่อนไข:", error_conditions)
else:
    print("ผ่านทุกเงื่อนไข")
