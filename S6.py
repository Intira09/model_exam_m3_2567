# pip install spacy_thai
# pip install python-crfsuite

import openai
import re
import time
import spacy_thai
from pythainlp.tokenize import sent_tokenize

# ---------------- โหลด spaCyThai ----------------
nlp = spacy_thai.load()

# ---------------- Typhoon API ----------------
client = openai.OpenAI(
    api_key= "", # add apikey
    base_url="https://api.opentyphoon.ai/v1"
)

# ---------------- ข้อมูลนักเรียน ----------------
raw_student_answer = """สื่อสังคมออนไลน์ ช่องทางไว้เผยแพร่กระจายข่าวสารไม่ว่าจะอยู่ส่วนไหนเราก็
สามารถสื่อสารกันได้ เราหาข้อมูลได้ มีประโยชน์เป็นอย่างมากและเราควรระมัดระวังในการ
ใช้สื่อสังคมออนไลน์ การพูดถึงผู้อื่นวิพากษ์วิจารณ์ต่าง ไม่จำเป็นผิดกฎหมาย เช่น การพนัน
การขายของผิดกฎหมาย ใช้อย่างระวังไม่ให้ตกเป็นเหยื่อมิจฉาชีพควรใช้ให้อย่างระมัดระวัง
เพราะสังคมออนไลน์เข้าถึงได้ง่ายมาก การซื้อ การพิมพ์ควรพิจารณาให้ดีก่อนทุกครั้ง"""

student_answer = re.sub(r'[\s\n\-]+', '', raw_student_answer).strip()

# ---------------- ฟังก์ชัน Extract SVO จาก spaCyThai ----------------
def extract_svo_spacythai(sentence, subject_keywords=None, object_keywords=None):
    if subject_keywords is None:
        subject_keywords = []
    if object_keywords is None:
        object_keywords = []

    doc = nlp(sentence)

    subject_list, verb_list, object_list = [], [], []

    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass", "FIXN" , "CCONJ", "SCONJ"]:
            subject_list.append(token.text)
        elif token.dep_ == "ROOT" or token.pos_ == "VERB":
            verb_list.append(token.text)
        elif token.dep_ in ["obj", "dobj", "iobj"]:
            object_list.append(token.text)

    # ---------------- เช็ค special case ----------------
    first_token = doc[0] if len(doc) > 0 else None
    subject_text = ", ".join(subject_list) if subject_list else "(ไม่พบ)"
    if subject_text == "(ไม่พบ)":
        if (first_token and first_token.pos_ in ["VERB", "SCONJ", "AUX"]) or any(kw in sentence for kw in subject_keywords):
            subject_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    object_text = ", ".join(object_list) if object_list else "(ไม่พบ)"
    if object_text == "(ไม่พบ)":
        if any(kw in sentence for kw in object_keywords):
            object_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    # ---------------- กรณีพิเศษเพิ่มเติม ----------------
    # ถ้าประธานเป็น "(ไม่พบแต่ไม่ถือว่าผิด)" และมีกริยา แต่กรรมไม่พบ ให้กรรมเป็น "(ไม่พบแต่ไม่ถือว่าผิด)"
    if subject_text == "(ไม่พบแต่ไม่ถือว่าผิด)" and verb_list and object_text == "(ไม่พบ)":
        object_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    return {
        "sentence_text": sentence,
        "subject": subject_text,
        "verb": ", ".join(verb_list) if verb_list else "(ไม่พบ)",
        "object": object_text
    }


# ---------------- ฟังก์ชันถาม Q2 แบบ retry ----------------
def ask_typhoon_q2_retry(system_prompt, question, document, wait_sec=3, max_attempts=5):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model="typhoon-v2.1-12b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{question}\n{document}"}
                ],
                temperature=0,
                max_tokens=1000
            )
            ans = response.choices[0].message.content.strip()
            if ans:
                return ans
            else:
                print(f"⚠️ ไม่มีผลลัพธ์จาก Typhoon Q2 (retry {attempt})")
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาด Q2: {e} (retry {attempt})")
        time.sleep(wait_sec)
    return "(ไม่มีคำตอบจาก Typhoon Q2)"

# ---------------- Keyword ----------------
subject_keywords = ["สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์",
                    "สื่อ", "การ", "ตลอดจน", "ไม่ว่าจะเป็น", "ในชีวิตประจำวัน", "ทุกวัย",
                    "และ", "หรือ", "อีกทั้งยัง", "อย่างไรก็ตาม", "แต่"]

object_keywords = ["ข้อมูลข่าวสาร", "ข้อมูล", "ข่าวสาร", "แก่สังคม", "จำนวนมาก",
                   "สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์", "ทุกวัย"]

# ---------------- Q1 (ใช้ pythainlp + spacythai) ----------------
sentences = sent_tokenize(raw_student_answer)  # ตัดประโยค
svo_result = [extract_svo_spacythai(sent, subject_keywords, object_keywords) for sent in sentences]

# ---------------- Q2 ----------------
q2 = "จากประโยคที่ให้หาประโยคที่ไม่มีความหมาย ถ้าไม่มีตอบว่า ไม่มีประโยคที่ไม่สื่อความหมาย"
system_q2 = "คุณคือผู้เชี่ยวชาญด้านภาษาไทย ห้ามคิดคำเอง"
ans2 = ask_typhoon_q2_retry(system_q2, q2, raw_student_answer)

# ---------------- คำนวณคะแนน ----------------
score = 1.0  # สมมติเต็ม 2 คะแนน

# Q1: หัก 0.5 ต่อคำ S/V/O ที่ไม่พบ
for item in svo_result:
    for key in ['subject', 'verb', 'object']:
        if "(ไม่พบ)" in item[key]:
            score -= 0.5

# Q2: ถ้าไม่ได้เริ่มด้วย "ไม่มี" ให้หัก 0.5
if not ans2.strip().startswith("ไม่มี"):
    score -= 0.5

score = max(score, 0)

# ---------------- แสดงผล ----------------
print("=== Q1: SVO ===")
for idx, item in enumerate(svo_result, 1):
    print(f"ประโยคที่ {idx}: {item['sentence_text']}")
    print(f"  ประธาน: {item['subject']}")
    print(f"  กริยา: {item['verb']}")
    print(f"  กรรม: {item['object']}")
    print("-----")

print("\n=== Q2 ===")
print(ans2)

print(f"\nคะแนนรวม Q1+Q2: {score}")
