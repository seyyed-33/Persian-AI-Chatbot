import streamlit as st
import base64
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ============================================
# 1. تنظیمات صفحه
# ============================================
st.set_page_config(
    page_title="چت‌بات فارسی",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# 2. تابع پس‌زمینه (روش استاندارد برای آنلاین)
# ============================================
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
                background-repeat: no-repeat;
            }}
            [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}
            footer {{ visibility: hidden !important; }}
            .block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 800px !important; }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("عکس پس‌زمینه پیدا نشد. لطفا فایل background.jpg را در کنار app.py قرار دهید.")

# اسم فایل عکس رو اینجا ساده گذاشتم
add_bg_from_local('background.jpg')

# ============================================
# 3. استایل‌های اصلی
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap');

* { font-family: 'Vazirmatn', sans-serif !important; }
h1, h2, h3 { font-family: 'Vazirmatn', sans-serif !important; font-weight: 700 !important; }

/* حباب پیام‌ها (شیشه‌ای) */
[data-testid="stChatMessageContent"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    color: white !important;
    padding: 12px 16px !important;
    font-size: 16px !important;
    line-height: 1.8 !important;
}

/* حذف پس‌زمینه پیش‌فرض Streamlit */
[data-testid="stChatMessage"] { background: transparent !important; }

/* ورودی شیشه‌ای */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    margin-bottom: 20px !important;
}
[data-testid="stBottom"] > div { background: transparent !important; padding-bottom: 0px !important; }
[data-testid="stBottom"] { background: transparent !important; }

textarea, input, button { font-family: 'Vazirmatn', sans-serif !important; }
h1, h2, h3, p, .stMarkdown { color: white !important; }

/* چسباندن متن به آیکون‌ها */
[data-testid="column"] { padding: 0px !important; margin: 0px !important; }
[data-testid="stHorizontalBlock"] { gap: 5px !important; align-items: center !important; }
[data-testid="column"] .stMarkdown { margin-top: 0px !important; margin-bottom: 0px !important; }

/* تنظیم فاصله عنوان */
h1 { text-align: center !important; margin-bottom: 5px !important; margin-top: 30px !important; }
h1 + div p, .stMarkdown p:has(+ div) { text-align: center !important; margin-bottom: 60px !important; font-size: 16px !important; opacity: 0.8 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 4. عنوان
# ============================================
st.markdown("<h1 style='text-align: center; margin-bottom: 0px;'>چت‌بات فارسی</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 50px; opacity: 0.8;'>برای مکالمه های ساده</p>", unsafe_allow_html=True)

# ============================================
# 5. مدیریت حافظه
# ============================================
msgs = StreamlitChatMessageHistory(key="chat_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("سلام خوش اومدی چطور میتونم کمکتون کنم")

# ============================================
# 6. تنظیمات LLM و پرامپت
# ============================================
llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=1000
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "تو یک چت بات هستی با کاربر مثل یک دوست رفتار کن و فارسی و با ادب جوابش رو بده از تاریخچه چت برای درک بهتر استفاده کن."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# ============================================
# 7. زنجیره با حافظه
# ============================================
chain = prompt | llm | StrOutputParser()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# ============================================
# 8. نمایش تاریخچه چت
# ============================================
for msg in msgs.messages:
    if msg.type == "ai":
        st.markdown(f"""
        <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; margin-bottom: 15px; direction: rtl;">
            <div style="background-color: #FFFFFF; min-width: 35px; height: 35px; border-radius: 50%; margin-left: 10px;"></div>
            <div style="flex-grow: 1; text-align: right; color: white; line-height: 1.8;">{msg.content}</div>
        </div>
        """, unsafe_allow_html=True)
            
    elif msg.type == "human":
        st.markdown(f"""
        <div style="display: flex; flex-direction: row; align-items: center; margin-bottom: 15px; direction: ltr;">
            <div style="background-color: #1E3A8A; min-width: 35px; height: 35px; border-radius: 50%; margin-right: 10px;"></div>
            <div style="flex-grow: 1; text-align: left; color: white; line-height: 1.8;">{msg.content}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# 9. ورودی کاربر جدید
# ============================================
if user_input := st.chat_input("پیام خود را وارد کنید"):
    
    st.markdown(f"""
    <div style="display: flex; flex-direction: row; align-items: center; margin-bottom: 15px; direction: ltr;">
        <div style="background-color: #1E3A8A; min-width: 35px; height: 35px; border-radius: 50%; margin-right: 10px;"></div>
        <div style="flex-grow: 1; text-align: left; color: white; line-height: 1.8;">{user_input}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("loading..."):
        try:
            response = chain_with_history.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": "any"}}
            )
            st.markdown(f"""
            <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: center; margin-bottom: 15px; direction: rtl;">
                <div style="background-color: #FFFFFF; min-width: 35px; height: 35px; border-radius: 50%; margin-left: 10px;"></div>
                <div style="flex-grow: 1; text-align: right; color: white; line-height: 1.8;">{response}</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.error("مشکل دارد یا فیلتر شکن شما قطع شده است API")