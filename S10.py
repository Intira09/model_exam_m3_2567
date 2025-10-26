# pip install pythainlp

import requests
from pythainlp.tokenize import word_tokenize

# ---------- ตั้งค่า ----------
TNER_API_KEY = '' # add api
CYBERBULLY_API_KEY = '' # add api

personal_pronoun_1 = {"หนู", "ข้า", "กู"}
personal_pronoun_2 = {"คุณ", "แก", "เธอ", "ตัวเอง", "เอ็ง", "มึง"}
all_personal_pronouns = personal_pronoun_1.union(personal_pronoun_2)

def check_named_entities(text):
    url = "https://api.aiforthai.in.th/tner"
    headers = {"Apikey": TNER_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            ner_result = response.json()
            bad_tags = {'ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB'}
            bad_entities = [ent['word'] for ent in ner_result.get("entities", []) if ent['tag'] in bad_tags]
            if bad_entities:
                return True, bad_entities
    except:
        pass
    return False, []

def check_cyberbully(text):
    url = "https://api.aiforthai.in.th/cyberbully"
    headers = {"Apikey": CYBERBULLY_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            result = response.json()
            if result.get("bully", "no") == "yes":
                bully_words = result.get("bully_words") or result.get("bully_phrases") or [text]
                return True, bully_words
    except:
        pass
    return False, []

def check_personal_pronouns(text):
    tokens = word_tokenize(text, engine="newmm")
    found_pronouns = [token for token in tokens if token in all_personal_pronouns]
    if found_pronouns:
        return True, found_pronouns
    return False, []

# ✅ ฟังก์ชันตรวจข้อความ 1 ข้อ
def evaluate_answer(text):
    mistake_count = 0
    mistakes = []

    ne_flag, ne_words = check_named_entities(text)
    if ne_flag:
        mistake_count += 1
        mistakes.append(f"มีชื่อเฉพาะ: {', '.join(ne_words)}")

    bully_flag, bully_words = check_cyberbully(text)
    if bully_flag:
        mistake_count += 1
        mistakes.append(f"ข้อความ Cyberbully: {', '.join(bully_words)}")

    pronoun_flag, pronouns = check_personal_pronouns(text)
    if pronoun_flag:
        mistake_count += 1
        mistakes.append(f"มีสรรพนามบุรุษที่ 1/2: {', '.join(pronouns)}")

    # กำหนดคะแนน
    if mistake_count == 0:
        score = 2
    elif mistake_count == 1:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "mistakes": mistakes or ["ไม่มีข้อผิดพลาด"]
    }

text = "มึงมันโง่ หนูไม่ชอบเลย"
result = evaluate_answer(text)

print("คะแนน:", result["score"])
print("ข้อผิดพลาด:", result["mistakes"])
