import os
import base64
import json
import io
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# .env dosyasından anahtarı çek ve Gemini'yi kur
# Önce Streamlit sunucusuna bak, bulamazsa lokaldeki .env dosyana bak
load_dotenv()
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

WRONG_QUESTIONS_PATH = "wrong_questions.json"
EXAMS_PATH = "exam_results.json"

# --- YARDIMCI FONKSİYONLAR ---

def load_wrong_questions() -> List[Dict[str, Any]]:
    if not os.path.exists(WRONG_QUESTIONS_PATH): return []
    try:
        with open(WRONG_QUESTIONS_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def append_wrong_question(record: Dict[str, Any]) -> None:
    data = load_wrong_questions()
    data.append(record)
    with open(WRONG_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_exam_results() -> List[Dict[str, Any]]:
    if not os.path.exists(EXAMS_PATH): return []
    try:
        with open(EXAMS_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def append_exam_result(record: Dict[str, Any]) -> None:
    data = load_exam_results()
    data.append(record)
    with open(EXAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_weekly_wrong_analysis() -> Optional[Dict[str, Any]]:
    data = load_wrong_questions()
    if not data: return None
    df = pd.DataFrame(data)
    if "date" not in df.columns: return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    now = datetime.now()
    start_of_week = now - timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    df_week = df[df["date"] >= start_of_week]
    if df_week.empty: return None
    return {
        "df": df_week,
        "topic_counts": df_week["topic"].value_counts(),
        "lesson_counts": df_week["lesson"].value_counts(),
        "start_of_week": start_of_week.date(),
        "end_of_week": now.date(),
    }

@st.cache_data
def load_data(csv_path: str = "veri.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)

def parse_net_range(value: Any) -> Optional[float]:
    if not isinstance(value, str): return None
    v = value.strip().replace(" ", "")
    if "-" in v:
        parts = v.split("-")
        try: return sum(float(p.replace("+", "")) for p in parts if p) / len(parts)
        except ValueError: return None
    if v.endswith("+"):
        try: return float(v[:-1])
        except ValueError: return None
    try: return float(v)
    except ValueError: return None

def parse_hour_range(value: Any) -> float:
    if not isinstance(value, str): return 0.0
    v = value.strip().replace(" ", "")
    if "-" in v:
        parts = v.split("-")
        try: return sum(float(p) for p in parts if p) / len(parts)
        except ValueError: return 0.0
    return float(v.replace("+", "")) if "+" in v else float(v) if v.replace(".","").isdigit() else 0.0

def find_closest_profiles(df, user_tyt, user_ayt, user_hours, user_style, top_k=3):
    df = df.copy()
    df["tyt_start_num"] = df["Sene başında TYT netlerin hangi aralıktaydı?"].apply(parse_net_range)
    df["ayt_start_num"] = df["İlk denemelerde AYT netlerin hangi aralıktaydı?"].apply(parse_net_range)
    hour_col = "Günlük ortalama çalışma saatin (Verimli geçen süre)?"
    df["hour_avg"] = df[hour_col].apply(parse_hour_range) if hour_col in df.columns else 0.0
    df = df.dropna(subset=["tyt_start_num", "ayt_start_num"])
    if df.empty: return df
    def calculate_score(row):
        net_dist = (abs(row["tyt_start_num"] - user_tyt)**2 + abs(row["ayt_start_num"] - user_ayt)**2)**0.5
        hour_penalty = abs(row["hour_avg"] - user_hours) * 3.0 if user_hours > 0 else 0.0
        style_col = "Çalışma stilini en iyi hangisi tanımlar?"
        style_bonus = 15.0 if pd.notna(row.get(style_col)) and user_style and str(user_style).lower() in str(row[style_col]).lower() else 0.0
        return net_dist + hour_penalty - style_bonus
    df["distance_score"] = df.apply(calculate_score, axis=1)
    return df.sort_values("distance_score").head(top_k)

# --- PROMPT VE CHAT MANTIĞI ---

def build_prompt(user_info: Dict[str, Any], profiles: List[Dict[str, Any]]) -> str:
    exam_date = datetime(2026, 6, 20))
    days_left = (exam_date - datetime.now()).days
    kronotip = user_info.get("kronotip", "Gündüzcü")
    
    bio_clock = (
        "Öğrenci GÜNDÜZCÜ: Sabahları erken başlamayı seviyor (Örn: 06:00-08:00 arası başlayabilir). "
        if "Gündüz" in kronotip else 
        "Öğrenci GECECİ: Gece geç saatlere kadar (01:00-02:00) çalışabilir. Sabahları çok erken başlatma."
    )

    lines = [
        "Sen profesyonel bir YKS koçusun. Öğrenciyle samimi ama disiplinli bir dille konuş.",
        f"SINAVA KALAN SÜRE: {days_left} gün. Buna göre deneme/konu dengesini kur.",
        f"BİYOLOJİK SAAT: {bio_clock}",
        "ÖNEMLİ: Saatleri katı verme, esneklik payı bırak. 'Öğlen Bloğu' veya '16:00 civarı' gibi ifadeler kullan.",
        "Derslerin yanına parantez içinde o gün neye odaklanması gerektiğini yaz.",
        "\n=== ÖĞRENCİ BİLGİLERİ ==="
    ]
    for k, v in user_info.items(): lines.append(f"{k}: {v}")
    
    if profiles:
        lines.append("\n=== BENZER BAŞARI HİKAYELERİ ===")
        for i, p in enumerate(profiles, start=1):
            lines.append(f"\n--- Profil {i} ---")
            for k, v in p.items():
                if k not in ["tyt_start_num", "ayt_start_num", "hour_avg", "distance_score"]:
                    lines.append(f"{k}: {v}")

    lines.extend(["\nFORMAT: Önce kısa bir motivasyon cümlesi, sonra stratejin, en son haftalık program."])
    return "\n".join(lines)

def call_gemini(user_info, profile_df):
    if not api_key: raise RuntimeError("API KEY eksik.")
    prompt = build_prompt(user_info, profile_df.to_dict(orient="records") if not profile_df.empty else [])
    model = genai.GenerativeModel('gemini-2.5-flash')
    return model.generate_content(prompt).text

def call_gemini_chat(user_info, current_schedule, chat_history):
    """Program oluştuktan sonraki sohbet botu altyapısı"""
    if not api_key: raise RuntimeError("API KEY eksik.")
    
    system_instruction = f"""Sen deneyimli, samimi ve disiplinli bir YKS koçusun.
    Öğrenci Profili: {user_info}
    Mevcut Program: {current_schedule}
    
    Görevlerin:
    1. Öğrenci sohbet ediyorsa ("Selam", "Nasılsın", "Sıkıldım" vb.) doğal, motive edici ve bir koç gibi yanıt ver.
    2. Öğrenci programda değişiklik istiyorsa, eski programı esneterek güncel halini sun.
    3. Sorularına YKS odağında taktiksel cevaplar ver. Her şeye sadece tablo çizerek cevap verme."""
    
    gemini_history = []
    # Son mesajı prompt olarak atacağımız için onu history'den hariç tutuyoruz
    for msg in chat_history[:-1]:
        if msg["role"] == "system": continue
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [str(msg["content"])]})
        
    model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=system_instruction)
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(str(chat_history[-1]["content"]))
    return response.text

def call_gemini_solve_image(prompt_text, image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    model = genai.GenerativeModel('gemini-2.5-flash')
    full_prompt = f"Sen bir YKS hocasısın. Soruyu kısa ve net çöz. LaTeX kullanma.\nNot: {prompt_text}"
    return model.generate_content([full_prompt, img]).text

# --- STREAMLIT UI ---

def init_session_state():
    if "messages" not in st.session_state: st.session_state.messages = []
    if "answers" not in st.session_state: st.session_state.answers = {}
    if "q_index" not in st.session_state: st.session_state.q_index = 0
    if "schedule" not in st.session_state: st.session_state.schedule = None

QUESTIONS = [
    {"key": "hedef_puan", "text": "Hedef YKS sıralaman nedir?", "type": "float"},
    {"key": "tyt_net", "text": "Şu anki TYT netin?", "type": "float"},
    {"key": "ayt_net", "text": "Şu anki AYT netin?", "type": "float"},
    {"key": "gun_sayisi", "text": "Haftada kaç gün çalışabilirsin?", "type": "int"},
    {"key": "saat_sayisi", "text": "Günlük ortalama kaç saat?", "type": "float"},
    {"key": "kronotip", "text": "Biyolojik saatin hangisi? 'Gündüzcü' mü, 'Gececi' mi?", "type": "text"},
    {"key": "zayif_konular", "text": "Zayıf derslerin/konuların?", "type": "text"},
    {"key": "okul_saatleri", "text": "Okul/Dershane saatlerin?", "type": "text"},
]

def parse_answer(raw, q_type):
    raw = raw.strip()
    if q_type == "float": return float(raw.replace(",", "."))
    if q_type == "int": return int(raw)
    return raw

def main():
    st.set_page_config(page_title="AI YKS Koçu", page_icon="🎓", layout="wide")
    init_session_state()
    
    try:
        df = load_data("veri.csv")
        data_loaded = True
    except:
        df = pd.DataFrame(); data_loaded = False

    with st.sidebar:
        st.title("🎓 YKS AI Koçu")
        secim = st.radio("Menü", ["🧠 Program Yap", "📸 Soru Çöz", "📊 Deneme Takibi", "📈 Analiz"])
        if api_key: st.success("API: Aktif")

    if secim == "🧠 Program Yap":
        st.header("🧠 Akıllı Program ve Sohbet Botu")
        
        # Chat geçmişini göster
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        # ANKET AŞAMASI
        if st.session_state.q_index < len(QUESTIONS):
            if not st.session_state.messages:
                st.session_state.messages.append({"role": "assistant", "content": "Selam! Programını hazırlayalım. " + QUESTIONS[0]["text"]})
                st.rerun()

            u_input = st.chat_input("Yanıtını yaz...")
            if u_input:
                st.session_state.messages.append({"role": "user", "content": u_input})
                idx = st.session_state.q_index
                
                # Arkadaşının çökerttiği yeri düzelten hata yakalama (Try-Except)
                try:
                    parsed_val = parse_answer(u_input, QUESTIONS[idx]["type"])
                    st.session_state.answers[QUESTIONS[idx]["key"]] = parsed_val
                    st.session_state.q_index += 1
                    
                    if st.session_state.q_index < len(QUESTIONS):
                        st.session_state.messages.append({"role": "assistant", "content": QUESTIONS[st.session_state.q_index]["text"]})
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "Bilgiler tamam! Aşağıdaki butona basarak programını oluşturabilirsin."})
                except ValueError:
                    st.session_state.messages.append({"role": "assistant", "content": "Bunu anlayamadım 😅 Sayısal bir cevap bekliyordum, sadece sayı yazarak tekrar dener misin?"})
                st.rerun()
        
        # PROGRAM OLUŞTURMA VE SERBEST SOHBET
        else:
            if not st.session_state.schedule:
                if st.button("🚀 Programımı Oluştur", type="primary"):
                    ans = st.session_state.answers
                    closest = find_closest_profiles(df, ans["tyt_net"], ans["ayt_net"], ans["saat_sayisi"], "pratik")
                    st.session_state.schedule = call_gemini(ans, closest)
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.schedule})
                    st.rerun()
            else:
                # SERBEST SOHBET (CHATBOT MODU)
                chat_input = st.chat_input("Programa dair bir şey iste veya sohbet et (Örn: Selam, Cuma gününü boşalt)")
                if chat_input:
                    st.session_state.messages.append({"role": "user", "content": chat_input})
                    with st.chat_message("user"): st.markdown(chat_input)
                    with st.spinner("Koçun yanıtlıyor..."):
                        try:
                            reply = call_gemini_chat(st.session_state.answers, st.session_state.schedule, st.session_state.messages)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Bir hata oluştu: {e}")

    elif secim == "📸 Soru Çöz":
        st.header("📸 Görsel Soru Çözümü")
        up = st.file_uploader("Soru fotoğrafı", type=["png", "jpg", "jpeg"])
        if st.button("Çöz") and up:
            with st.spinner("İnceleniyor..."):
                res = call_gemini_solve_image("Çöz", up.getvalue())
                st.markdown(res)

    elif secim == "📊 Deneme Takibi":
        st.header("📊 Net Kaydı")
        with st.form("net_form"):
            t_net = st.number_input("TYT Toplam", step=0.25)
            a_net = st.number_input("AYT Toplam", step=0.25)
            if st.form_submit_button("Kaydet"):
                append_exam_result({"timestamp": datetime.now().isoformat(), "tyt_total": t_net, "ayt_total": a_net})
                st.success("Kaydedildi!")

    elif secim == "📈 Analiz":
        st.header("📈 Haftalık Gelişim")
        an = get_weekly_wrong_analysis()
        if an:
            st.bar_chart(an["lesson_counts"])
            st.write(f"Bu hafta {len(an['df'])} soru çözüldü.")
        else: st.info("Henüz veri yok.")

if __name__ == "__main__":
    main()
