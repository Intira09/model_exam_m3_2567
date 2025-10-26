import re
import pandas as pd
import requests
from transformers import pipeline
import json

# -------------------------
# โหลดโมเดล mask-filling
# -------------------------
fill_mask = pipeline("fill-mask", model="xlm-roberta-base", tokenizer="xlm-roberta-base")

# -------------------------
# API Key สำหรับ T-NER
# -------------------------
API_KEY = "" # add api

def call_tner(text):
    url = "https://api.aiforthai.in.th/tner"
    headers = {"Apikey": API_KEY}
    try:
        response = requests.post(url, headers=headers, data={"text": text}, timeout=10)

        # เช็คว่าคำตอบเป็น JSON หรือไม่
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print("❌ Response ไม่ใช่ JSON:", response.text[:200])
                return {}
        else:
            print(f"❌ API Error {response.status_code}: {response.text[:200]}")
            return {}
    except requests.exceptions.RequestException as e:
        print("❌ Request error:", e)
        return {}


# -------------------------
# ฟังก์ชันดึงคำตาม POS
# -------------------------
def find_words_by_pos(tner_result, pos_tags):
    words = []
    tokens = tner_result.get("words", [])
    pos = tner_result.get("POS", [])
    for idx, (w, p) in enumerate(zip(tokens, pos)):
        if p in pos_tags:
            words.append((idx, w))
    return words

# -------------------------
# ฟังก์ชัน fill-mask
# -------------------------
def check_word_with_fill_mask(sentence, target_word):
    masked_sentence = sentence.replace(target_word, fill_mask.tokenizer.mask_token, 1)
    preds = fill_mask(masked_sentence)
    predicted_tokens = [p["token_str"].strip() for p in preds]
    return True, predicted_tokens

# -------------------------
# ฟังก์ชัน normalize (เอา whitespace และ newlines ออก)
# -------------------------
def normalize_word(w):
    return w.replace("\n", "").replace("\r", "").replace(" ", "").lower()

# -------------------------
# โหลด dataset และแปลงเป็น set
# -------------------------
spoken_words_dataset = pd.read_csv("/content/dataset_speak_word(in).csv")["word"].dropna().tolist()
notinlan_dataset = pd.read_csv("/content/dataset_notinlan_words(in).csv")["notinlan"].dropna().tolist()
local_words_context = pd.read_csv("/content/S12_sample_local_dialect  (1)(in)(in).csv")["local_word"].dropna().tolist()

spoken_words_set = set(normalize_word(w) for w in spoken_words_dataset)
notinlan_set = set(normalize_word(w) for w in notinlan_dataset)
local_dialect_set = set(normalize_word(w) for w in local_words_context)

# -------------------------
# keyword_dict ตัวอย่าง
# -------------------------
keyword_dict = {
      "conjunctions": ["จน", "แม้มี", "แม้ว่า", "ถ้าใช้", "จึงมี", "และเรา", "ดังนั้น", "รับผิดชอบ", "ระวัง", "วันต่อมา", "เช่น การเตือนภัย",
                       "หากใช้", "แต่ปัจจุบัน", "อย่างไรก็ตาม", "อย่างไม่", "หรือ", "ถึงแม้", "ยังมี", "การ", "เพราะ",
                       "ในทั้งนี้", "เพื่อเป็น", "เพื่อความ", "แม้แต่", "ตลอดจน", "ถ้าหาก", "เพราะฉะนั้น", "ส่วน", "บุคคล", "เข้าถึง",
                       "ข้อมูลข่าวสาร", "ทำให้", "ไม่งั้น", "โทษแก่สังคม", "หลายคน", "แก้ปัญหา",
                       "สะดวก", "สามารถ", "ระดม", "การสร้าง", "จะเป็น", "ไปเจอ", "จำเป็น", "ค้าขาย",
                       "ตัวอย่าง", "หรือติ"],

      "prepositions": ["ในการ", "ในทางที่ดี", "ให้กับ", "ด้วย", "ของสังคม", "ของข้อมูล", "ต่อสังคม", "ยัง", "อย่างรวดเร็ว",
                       "ก็จะ", "เมื่อพบว่า", "แก่สังคม", "แก่ผู้อื่น", "ลักษณะดังกล่าว", "จากคนบางกลุ่ม", "กิจการของตน", "สื่อ",
                       "ทั่วโลก", "ทางด้าน", "ต่อง", "รวมถึง", "เกี่ยวกับ", "พอ", "โรงงาน", "ฉะนั้น", "เป็นโทษ",
                       "ช่วยเหลือ", "เดือดร้อน", "มีการ", "อีกด้วย", "ชวนเชื่อ", "ก่อนหลงเชื่อ", "ไม่ตรง",
                       "แต่หลายคน", "ชีวิตประจำวัน", "สะดวก", "ข้อมูลข่าวสาร", "ติดต่อผ่าน", "สำหรับ",
                       "เรื่องต่างๆ", "เรียกได้เลย", "เข้าถึง", "กันเป็นจำนวนมาก", "การขาย", "การสื่อสาร",
                       "ทุกคน", "ส่วนใหญ่", "ตัวเอง", "ทางการ", "แทนการ"],

      "classifiers": ["หลายครั้ง", "บางคน", "ทุกวัน", "เหล่านี้", "อย่างมาก", "สำคัญ", "ไม่ดี", "อาจจะ",
                      "ไม่ว่าจะเป็น", "เสียใจมาก", "เข้าถึง", "ได้เลย"]
}

# -------------------------
# ฟังก์ชันประเมินข้อความนักเรียน (เวอร์ชันปรับปรุง)
# -------------------------
def evaluate_student_text(student_text, keyword_dict,
                          spoken_words_set,
                          notinlan_set,
                          local_dialect_set,
                          full_score=2.0,
                          deduct_per_word=0.5):
    """
    ตรวจข้อความนักเรียน
    - ตรวจ POS (conjunctions, prepositions, classifiers) + fill-mask
    - ตรวจคำภาษาพูด / คำไม่มีในภาษา / คำถิ่น
    - คืน errors และคะแนน
    """
    errors = {k: [] for k in (list(keyword_dict.keys()) + ["slang", "dialect", "invalid_word"])}
    total_wrong = 0
    penalty = 0.0

    # 1️⃣ ตรวจ POS + fill-mask
    tner_result = call_tner(student_text)
    words_list = [w.strip() for w in re.split(r'\s+', student_text.strip()) if w.strip()]
    pos_categories = {
        "conjunctions": ["CNJ"],
        "prepositions": ["P"],
        "classifiers": ["CL"]
    }

    for key, pos_list in pos_categories.items():
        words = find_words_by_pos(tner_result, pos_list)
        for idx, word in words:
            prev_word = words_list[idx-1] if 0 <= idx-1 < len(words_list) else ""
            next_word = words_list[idx+1] if 0 <= idx+1 < len(words_list) else ""

            is_valid, preds = check_word_with_fill_mask(student_text, word)

            # normalize word เอา \n ออกก่อนตรวจ keyword
            clean_word = normalize_word(word)
            clean_prev = normalize_word(prev_word)
            clean_next = normalize_word(next_word)

            matched_keyword = None
            for kw in keyword_dict.get(key, []):
                if kw.startswith(clean_prev + clean_word + clean_next) or clean_word in kw:
                    matched_keyword = kw
                    break

            is_wrong = False
            if preds:
                if not matched_keyword and (clean_word not in preds):
                    is_wrong = True

            if is_wrong:
                total_wrong += 1

            errors[key].append({
                "word": clean_word,
                "predicted": preds,
                "prev_word": prev_word,
                "next_word": next_word,
                "matched_keyword": matched_keyword,
                "is_wrong": is_wrong
            })

    # 2️⃣ ตรวจคำภาษาพูด / คำไม่มีในภาษา / คำถิ่น
    clean_text = normalize_word(student_text)

    # ภาษาพูด
    spoken_words_found = [w for w in spoken_words_set if w in clean_text]
    for w in spoken_words_found:
        errors["slang"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    # คำไม่มีในภาษา
    notinlan_found = [w for w in notinlan_set if w in clean_text]
    for w in notinlan_found:
        errors["invalid_word"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    # คำถิ่น
    dialect_found = [w for w in local_dialect_set if w in clean_text]
    for w in dialect_found:
        errors["dialect"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    # 3️⃣ คำนวณคะแนนรวม
    score = full_score - 0.5 * total_wrong - penalty
    score = max(min(score, full_score), 0.0)

    return {
        "errors": errors,
        "score": round(score,2)
    }

# -------------------------
# 🔥 ตัวอย่างการใช้งาน
# -------------------------
if __name__ == "__main__":
    student_answer = """เห็นด้วย เพราะยุคสมัยนี้ล้วนมีแต่การใช้สื่อสังคมออนไลน์อย่างชัดเจน
มีทั้งการใช้ไปในทางที่ดีและไม่ดีการสร้างความเสื่อมเสียให้ผู้อื่น การวิจารณ์
ใช้ให้เป็นประโยชน์สามารถสร้างสรรค์สิ่งที่ดีให้แก่สังคมได้ แต่บางคนหากไม่
ถูกใจก็ทำให้ผู้อื่นเสียหาย เขียนวิจารณ์ในทางที่ไม่ดี คือ การทำร้าย
กันทางอ้อม ไม่ตระหนักคิดถึงผลที่จะตามมาภายหลัง"""

    result = evaluate_student_text(student_answer, keyword_dict,
                                   spoken_words_set,
                                   notinlan_set,
                                   local_dialect_set)
    print(json.dumps(result, ensure_ascii=False, indent=2))
