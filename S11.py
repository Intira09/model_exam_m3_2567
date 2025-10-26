import re
import json
import difflib
import requests
from sentence_transformers import SentenceTransformer, util
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.tag import pos_tag
from pythainlp.util import normalize
import pandas as pd
from pythainlp.spell import spell

# ---- ส่วนให้คะแนนการสะกดคำ ----
# โหลด whitelist คำทับศัพท์
with open('/content/thai_loanwords_new_update(1).json', 'r', encoding='utf-8') as f:
     loanwords_data = json.load(f)
     loanwords_whitelist = set(item['thai_word'] for item in loanwords_data)

# โหลด common misspellings JSON
with open('/content/update_common_misspellings.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    COMMON_MISSPELLINGS = {item['wrong']: item['right'] for item in data}

# โหลด list คำที่สามารถฉีกคำได้
with open("/content/splitable_phrases.json", "r", encoding="utf-8") as f:
    splitable_phrases = set(json.load(f))  # ✅ ให้เป็น set เสมอ

API_KEY = '' # add Apikey
API_URL = 'https://api.longdo.com/spell-checker/proof'

custom_words = {"ประเทศไทย", "สถาบันการศึกษา", "นานาประการ"}

#คำที่ไม่สามารถฉีกคำได้
strict_not_split_words = { 'มากมาย', 'ประเทศไทย', 'ออนไลน์', 'ความคิดเห็น', 'ความน่าเชื่อถือ' }

thai_dict = set(w for w in set(thai_words()).union(custom_words) if (' ' not in w) and w.strip())

# allowed punctuation (เพิ่ม ' และ ")
allowed_punctuations = [ '.', ',', '-', '(', ')', '!', '?', '%', '“', '”', '‘', '’', '"', "'", '…', 'ฯ' ]

# Allow / Forbid list ไม้ยมก (เพิ่มคำที่ใช้บ่อย)
allow_list = {'ปี', 'อื่น', 'เล็ก', 'ใหญ่', 'มาก', 'หลาย', 'ช้า', 'เร็ว', 'ชัด', 'ดี', 'ผิด' ,'เสีย', 'หาย','สวย','มั่ว','ง่าย'}
forbid_list = {'นา', 'บางคน', 'บางอย่าง', 'บางสิ่ง', 'บางกรณี'}

# อนุญาตให้แยกกันได้
splitable_pairs = { ("ไป", "มา"),("ได้","ส่วน"),("ดี","แต่") }

#------------------------------------------------------------------------#
# #ตรวจการฉีกคำ
def check_linebreak_issue(prev_line_tokens, next_line_tokens, max_words=3):
    last_word = prev_line_tokens[-1]
    first_word = next_line_tokens[0]
    if last_word.endswith('-') or first_word.startswith('-'):
        return False, None, None, None
    for prev_n in range(1, min(max_words, len(prev_line_tokens)) + 1):
        prev_part = ''.join(prev_line_tokens[-prev_n:])
        for next_n in range(1, min(max_words, len(next_line_tokens)) + 1):
            next_part = ''.join(next_line_tokens[:next_n])
            combined = normalize(prev_part + next_part)
            if ( (' ' not in combined) and (combined not in splitable_phrases) and ( (combined in strict_not_split_words) or ( (combined in thai_dict) and (len(word_tokenize(combined, engine='newmm')) == 1) ) ) ):
                return True, prev_part, next_part, combined
    return False, None, None, None

#วนตรวจทั้งข้อความทีละบรรทัด
def analyze_linebreak_issues(text):
    lines = text.strip().splitlines()
    issues = []
    for i in range(len(lines) - 1):
        prev_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        prev_tokens = word_tokenize(prev_line)
        next_tokens = word_tokenize(next_line)
        if not prev_tokens or not next_tokens:
            continue
        issue, prev_part, next_part, combined = check_linebreak_issue(prev_tokens, next_tokens)
        if issue:
            issues.append({
                           'line_before': prev_line,
                           'line_after': next_line,
                           'prev_part': prev_part,
                           'next_part': next_part,
                           'combined': combined,
                           'pos_in_text': (i, len(prev_tokens)) })
    return issues

#รวมข้อความหรือคำที่ถูกตัดข้ามบรรทัด
def merge_linebreak_words(text, linebreak_issues):
    lines = text.splitlines()
    for issue in reversed(linebreak_issues):
        i, _ = issue['pos_in_text']
        lines[i] = lines[i].rstrip() + issue['combined'] + lines[i+1].lstrip()[len(issue['next_part']):]
        lines.pop(i+1)
    return "\n".join(lines) # Added return statement here

#ข้ามอังกฤษและตัวเลข
def is_english_or_number(word: str) -> bool:
    """ คืน True ถ้า word เป็นภาษาอังกฤษหรือตัวเลข """
    w = word.strip()
    # เช็คถ้าเป็นตัวอักษร A-Z, a-z, ตัวเลข 0-9 หรือมีแต่พวก .,()-_/ ปน
    return bool(re.fullmatch(r"[A-Za-z0-9().,\-_/]+", w))

# 1. ตรวจการสะกดคำด้วย PyThaiNLP + Longdo
# # -------------------------------
def pythainlp_spellcheck(tokens, pos_tags, dict_words, ignore_words):
    """ ตรวจสอบการสะกดคำด้วย PyThaiNLP คืน list ของ dict ที่มี 'word', 'pos', 'index' """
    mistakes = []
    for i, token in enumerate(tokens):
        # Skip empty tokens, numbers, and English words
        if not token or is_english_or_number(token):
            continue

        # Check if the word is in our custom dictionary
        if token in ignore_words:
            continue

        # Check if the word is in the general Thai dictionary
        if token in dict_words:
            continue
        # Use PyThaiNLP spell checker
        suggestions = spell(token)

        # If no suggestions and not in dictionaries, consider it a potential error
        if not suggestions:
            mistakes.append({
                             'word': token,
                             'pos': pos_tags[i][1] if i < len(pos_tags) else None,
                             'index': i, 'suggestions': suggestions })
    return mistakes

def spellcheck_before_tokenize(text):
    words = re.findall(r'[ก-๙]+', text) # ✅ ดึงเฉพาะคำไทย
    pos_tags = pos_tag(words, corpus='orchid')

    # filter ข้ามอังกฤษ/เลข อีกชั้น
    words = [w for w in words if not is_english_or_number(w)]

    pythai_errors = pythainlp_spellcheck(
                                         words, pos_tags,
                                         dict_words=thai_dict,
                                         ignore_words=custom_words )
    wrong_words = [e['word'] for e in pythai_errors]

    longdo_results = longdo_spellcheck_batch(wrong_words)
    spelling_errors_legit = [
                             {**e, 'suggestions': longdo_results.get(e['word'], [])}
                             for e in pythai_errors if e['word'] in longdo_results ]
    return spelling_errors_legit

#longdo spell checker
def longdo_spellcheck_batch(words):
    """
    ตรวจสอบคำผิดด้วย Longdo API แบบ batch
    คืน dict {word: [suggestions]}
    """
    results = {}
    headers = {"Content-Type": "application/json"}
    for word in words:
        try:
            payload = {
                "text": word,
                "api": API_KEY
            }
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                data = response.json()
                suggestions = []
                # Longdo จะส่งผลลัพธ์เป็น list ของคำ
                if "words" in data and data["words"]:
                    for w in data["words"]:
                        if "candidates" in w:
                            suggestions.extend(c["text"] for c in w["candidates"])
                results[word] = suggestions
            else:
                results[word] = []
        except Exception as e:
            results[word] = []
    return results

# ตรวจ common misspellings ก่อน tokenize (จากข้อความดิบ)
def check_common_misspellings_before_tokenize(text, misspelling_dict):
    """ text : ข้อความดิบ (ยังไม่ tokenize) misspelling_dict : dict จากไฟล์ JSON คืน list ของ dict ที่มี 'word', 'index', 'right' """
    errors = []
    for wrong, right in misspelling_dict.items():
        if wrong in text:
            for m in re.finditer(re.escape(wrong), text):
                errors.append({
                               "word": wrong,
                               "index": m.start(), # ตำแหน่งในข้อความดิบ
                               "right": right })
    return errors

# ตรวจ loanwords ก่อน tokenize
def check_loanword_before_tokenize(words, whitelist):
    mistakes = []
    for i, w in enumerate(words):
        if is_english_or_number(w):  # ข้ามอังกฤษ+ตัวเลข
            continue
        matches = difflib.get_close_matches(w, list(whitelist), n=1, cutoff=0.7)
        if matches and w not in whitelist:
            mistakes.append({
                'word': w,                  # เปลี่ยนจาก 'found'
                'index': i,
                'suggestions': [matches[0]] # เปลี่ยนจาก 'should_be'
            })
    return mistakes

#ตรวจการใช้เครื่องหมายที่ไม่อนุญาต
allowed_phrases = ["ฯลฯ"]

# เครื่องหมายที่มีหลายตัวประกอบ
def find_unallowed_punctuations(text):
    # ลบวลีที่อนุญาตออกก่อนตรวจ
    for phrase in allowed_phrases:
        text = text.replace(phrase, "")

    pattern = f"[^{''.join(re.escape(p) for p in allowed_punctuations)}a-zA-Z0-9ก-๙\\s]"
    return list(set(re.findall(pattern, text)))

#ใช้แยกไม้ยมกออกจากคำที่ติดกัน
def separate_maiyamok(text):
    return re.sub(r'(\S+?)ๆ', r'\1 ๆ', text)

#ตรวจการใช้ไม้ยมก
def analyze_maiyamok(tokens, pos_tags):
    results = []
    found_invalid = False
    VALID_POS = {'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV', 'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'}
    for i, token in enumerate(tokens):
        if token == 'ๆ':
            prev_idx = i - 1
            prev_word = tokens[prev_idx] if prev_idx >= 0 else None
            prev_tag = pos_tags[prev_idx][1] if prev_idx >= 0 else None
            if prev_word is None or prev_word == 'ๆ': verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
            elif prev_word in forbid_list: verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
            elif (prev_tag in VALID_POS) or (prev_word in allow_list): verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
            else: verdict = '❌ ไม่ควรใช้ไม้ยมok นอกจากกับคำนาม/กริยา/วิเศษณ์'
            context = tokens[max(0, i-2):min(len(tokens), i+3)]
            results.append({
                            'คำก่อนไม้ยมก': prev_word or '',
                            'POS คำก่อน': prev_tag or '',
                            'บริบท': ' '.join(context),
                            'สถานะ': verdict })
            if verdict.startswith('❌'): found_invalid = True
    return results, found_invalid

#ตรวจการแยกคำ
def detect_split_errors(tokens, custom_words=None, splitable_phrases=None):
    check_dict = set(thai_words()).union(custom_words or [])
    check_dict = {w for w in check_dict if (' ' not in w) and w.strip()}
    splitable_phrases = splitable_phrases or set()

    errors = []
    for i in range(len(tokens) - 1):
        combined = tokens[i] + tokens[i + 1]

        # ✅ ตรวจว่า combined อยู่ใน dict
        # ❌ แต่ไม่อนุญาตให้แยก ถ้าไม่อยู่ใน splitable_pairs
        if (combined in check_dict) and ((tokens[i], tokens[i+1]) not in splitable_pairs):
            errors.append({
                           "split_pair": (tokens[i], tokens[i+1]),
                           "suggested": combined })
    return errors

#วิเคราะห์เงื่อนไข
def evaluate_text(text):
    # -----------------------------
    # ✅ เช็คความยาวคำตอบก่อนเริ่มประเมิน
    # -----------------------------
    if not text or not text.strip():
        return {
            'linebreak_issues': [],
            'spelling_errors': [],
            'loanword_spell_errors': [],
            'punctuation_errors': [],
            'maiyamok_results': [],
            'split_errors': [],
            'reasons': ["ไม่มีคำตอบ"],
            'score': 0.0,
            'total_error_count': 0
        }

    # ✅ ถ้านับบรรทัดแล้ว ≤ 2 บรรทัดให้ 0
    lines = [l for l in text.strip().splitlines() if l.strip()] # Define lines here
    char_count = len(text.replace(" ", "").replace("\n", "")) # Calculate char count here
    if len(lines) <= 2 and char_count < 50:
        return {
            'linebreak_issues': [],
            'spelling_errors': [],
            'loanword_spell_errors': [],
            'punctuation_errors': [],
            'maiyamok_results': [],
            'split_errors': [],
            'reasons': [f"ตอบสั้นเกินไป ({len(lines)} บรรทัด, {char_count} ตัวอักษร)"],
            'score': 0.0,
            'total_error_count': 0
        }
    # -----------------------------
    # 1) จัดการตัดบรรทัด
    # -----------------------------
    linebreak_issues = analyze_linebreak_issues(text)
    corrected_text = merge_linebreak_words(text, linebreak_issues)

    # ✅ ลบ newline เพิ่มเติม (กรณี merge_linebreak_words ไม่ได้จับทุกเคส)
    corrected_text = corrected_text.replace("\n", "")

    # ✅ normalize ก่อนตรวจ
    corrected_text = normalize(corrected_text)

    # 2) ตรวจ common misspellings ก่อน tokenize
    json_misspells = check_common_misspellings_before_tokenize(corrected_text, COMMON_MISSPELLINGS)

    # 3) tokenize ปกติ
    tokens = [t for t in word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
              if not is_english_or_number(t)]
    pos_tags = pos_tag(tokens, corpus='orchid')

    # ตรวจ spelling ด้วย PyThaiNLP
    pythai_errors = pythainlp_spellcheck(tokens, pos_tags, dict_words=thai_dict, ignore_words=custom_words)

    # ตรวจ loanwords (สองชั้น)
    # -----------------------------

    # 1) ตรวจแบบดิบ (จับทั้งประโยค)
    raw_words = re.findall(r'[ก-๙]+', corrected_text)
    raw_loanword_errors = check_loanword_before_tokenize(raw_words, loanwords_whitelist)

    # 2) ตรวจแบบ tokenize (ละเอียดกว่า แต่ขึ้นกับ tokenizer)
    token_loanword_errors = check_loanword_before_tokenize(tokens, loanwords_whitelist)

    # รวมผล
    loanword_errors = raw_loanword_errors + token_loanword_errors

    # ตรวจ Longdo
    longdo_results = longdo_spellcheck_batch([e['word'] for e in pythai_errors])
    longdo_errors = [
        {
            **e,
            'suggestions': [str(s) for s in longdo_results.get(e['word'], []) if s]  # ✅ sanitize
        }
        for e in pythai_errors
    ]

    # รวม spelling errors + sanitize suggestions ให้เป็น string เสมอ
    all_spelling_errors = longdo_errors + [
        {
            "word": e["word"],
            "pos": None,
            "index": e["index"],
            "suggestions": [str(e["right"])] if e["right"] else [],
        }
        for e in json_misspells
    ] + [
        {
            "word": e["word"],
            "pos": None,
            "index": e["index"],
            "suggestions": [str(s) for s in e.get("suggestions", []) if s],
        }
        for e in loanword_errors
    ]

    # ตรวจ punctuation, maiyamok, split word
    punct_errors = find_unallowed_punctuations(corrected_text)
    maiyamok_results, has_wrong_maiyamok = analyze_maiyamok(tokens, pos_tags)
    split_errors = detect_split_errors(tokens, custom_words=custom_words)

    # -----------------------------
    # ✅ ใหม่: รวมคำผิดซ้ำเป็น 1
    unique_spelling_words = {e["word"] for e in all_spelling_errors}
    unique_split_errors = {e["suggested"] for e in split_errors}

    total_errors = (
        len(unique_spelling_words) +
        len(linebreak_issues) +
        len(unique_split_errors) +
        len(punct_errors) +
        sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))
    )

    # -----------------------------
    # รวม reasons
    # -----------------------------
    reasons = []
    if linebreak_issues:
        details = [f"{issue['prev_part']} + {issue['next_part']} → {issue['combined']}" for issue in linebreak_issues]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))
    if split_errors:
        details = [f"{e['split_pair'][0]} + {e['split_pair'][1]} → {e['suggested']}" for e in split_errors]
        reasons.append("พบการแยกคำผิด: " + "; ".join(details))
    if all_spelling_errors:
        error_words = [
            f"{e['word']} (แนะนำ: {', '.join(str(s) for s in e.get('suggestions', []) if s)})"
            for e in all_spelling_errors
        ]
        reasons.append("ตรวจเจอคำสะกดผิดหรือทับศัพท์ผิด: " + ", ".join(error_words))
    if punct_errors:
        reasons.append(f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}")
    wrong_desc = [x for x in maiyamok_results if x['สถานะ'].startswith('❌')]
    if wrong_desc:
        texts = [f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}" for x in wrong_desc]
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(texts))
    if not reasons:
        reasons.append("ไม่มีปัญหา")

    # -----------------------------
    # ✅ ให้คะแนนแบบ 5 ระดับ
    # -----------------------------
    if total_errors == 0:
        score = 2.0
    elif total_errors == 1:
        score = 1.5
    elif total_errors == 2:
        score = 1.0
    elif total_errors == 3:
        score = 0.5
    else:  # 4 ขึ้นไป
        score = 0.0

    # Ensure a dictionary is always returned
    return {
        'linebreak_issues': linebreak_issues,
        'spelling_errors': all_spelling_errors,
        'loanword_spell_errors': loanword_errors,
        'punctuation_errors': list(punct_errors),
        'maiyamok_results': maiyamok_results,
        'split_errors': split_errors,
        'reasons': reasons,
        'score': score,
        'total_error_count': total_errors,
        'char_count': char_count # Add char count to the return dictionary
    }

def evaluate_single_answer(answer_text):
    res = evaluate_text(str(answer_text))
    spelling_score = res["score"]   # ตอนนี้อยู่ในสเกล 0,0.5,1,1.5,2
    spelling_reason = res["reasons"]

    result = {
        "คะแนนการสะกดคำ (2 คะแนน)": spelling_score,
        "เงื่อนไขที่ผิด": spelling_reason,
        "คะแนนรวมทั้งหมด": spelling_score
    }
    return json.dumps(result, ensure_ascii=False, indent=2)

# -------CSV---------
df = pd.read_csv("/content/drive/MyDrive/dataset_S11_ใหม่(Sheet1).csv")

คะแนนสะกด_list = []
เงื่อนไขผิด_list = []

for i, row in df.iterrows():
    text = row["student_answer_2"]
    if not isinstance(text, str):
        text = ""

    result = evaluate_text(text)
    คะแนนสะกด_list.append(result["score"])
    เงื่อนไขผิด_list.append("; ".join(result.get("reasons", [])))  # ✅ แก้ตรงนี้

df["S11"] = คะแนนสะกด_list
df["เงื่อนไขที่ผิด"] = เงื่อนไขผิด_list

df.to_csv("/content/sample_data/score14_s11.csv", index=False, encoding="utf-8-sig")
print("✅ ประเมินเสร็จแล้ว เฉพาะคะแนนการสะกดคำ (S11)")
