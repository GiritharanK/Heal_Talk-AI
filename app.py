import streamlit as st
from streamlit import session_state
import json
import os
from gtts import gTTS
from transformers import pipeline
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import streamlit as st
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0

languages = {
            "English": "en",
            "Afrikaans": "af",
            "Albanian": "sq",
            "Amharic": "am",
            "Arabic": "ar",
            "Armenian": "hy",
            "Azerbaijani": "az",
            "Basque": "eu",
            "Belarusian": "be",
            "Bengali": "bn",
            "Bosnian": "bs",
            "Bulgarian": "bg",
            "Catalan": "ca",
            "Cebuano": "ceb",
            "Chichewa": "ny",
            "Chinese (simplified)": "zh-cn",
            "Chinese (traditional)": "zh-tw",
            "Corsican": "co",
            "Croatian": "hr",
            "Czech": "cs",
            "Danish": "da",
            "Dutch": "nl",
            "Esperanto": "eo",
            "Estonian": "et",
            "Filipino": "tl",
            "Finnish": "fi",
            "French": "fr",
            "Frisian": "fy",
            "Galician": "gl",
            "Georgian": "ka",
            "German": "de",
            "Greek": "el",
            "Gujarati": "gu",
            "Haitian creole": "ht",
            "Hausa": "ha",
            "Hawaiian": "haw",
            "Hebrew": "he",
            "Hindi": "hi",
            "Hmong": "hmn",
            "Hungarian": "hu",
            "Icelandic": "is",
            "Igbo": "ig",
            "Indonesian": "id",
            "Irish": "ga",
            "Italian": "it",
            "Japanese": "ja",
            "Javanese": "jw",
            "Kannada": "kn",
            "Kazakh": "kk",
            "Khmer": "km",
            "Korean": "ko",
            "Kurdish (kurmanji)": "ku",
            "Kyrgyz": "ky",
            "Lao": "lo",
            "Latin": "la",
            "Latvian": "lv",
            "Lithuanian": "lt",
            "Luxembourgish": "lb",
            "Macedonian": "mk",
            "Malagasy": "mg",
            "Malay": "ms",
            "Malayalam": "ml",
            "Maltese": "mt",
            "Maori": "mi",
            "Marathi": "mr",
            "Mongolian": "mn",
            "Myanmar (burmese)": "my",
            "Nepali": "ne",
            "Norwegian": "no",
            "Odia": "or",
            "Pashto": "ps",
            "Persian": "fa",
            "Polish": "pl",
            "Portuguese": "pt",
            "Punjabi": "pa",
            "Romanian": "ro",
            "Russian": "ru",
            "Samoan": "sm",
            "Scots gaelic": "gd",
            "Serbian": "sr",
            "Sesotho": "st",
            "Shona": "sn",
            "Sindhi": "sd",
            "Sinhala": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Spanish": "es",
            "Sundanese": "su",
            "Swahili": "sw",
            "Swedish": "sv",
            "Tajik": "tg",
            "Tamil": "ta",
            "Telugu": "te",
            "Thai": "th",
            "Turkish": "tr",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uyghur": "ug",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Welsh": "cy",
            "Xhosa": "xh",
            "Yiddish": "yi",
            "Yoruba": "yo",
            "Zulu": "zu",
        }

load_dotenv()
genai.configure(api_key="AIzaSyArFsF8XTEyuPDbQhtvGjZfygziLN6RF7o")

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'], round(result['score'] * 100, 2)

def play(text, lang):
    try:
        speech = gTTS(text = text,lang=lang,slow = False)
        speech.save('audio.mp3')
        # audio_file = Path().cwd() /   'audio.mp3'
        # playsound("audio.mp3")
        if os.path.exists('audio.mp3'):
            os.remove('audio.mp3')
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def generate_response(text):
    prompt = f'''You are a helpful chatbot for mental health support. Respond to the ffollowing accordingly and empathetically  {text}'''
    try:
        response = model.generate_content(prompt)
        if len(response.text) == 0:
            return "I'm sorry, I don't understand. Can you please rephrase?"
        return response.text
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, couldn't generate a response"

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def translate(text, source_language, target_language):
    texts = []
    while len(text) > 5000:
        index = 5000
        while text[index] != "." and index > 0:
            index -= 1
        if index == 0:
            index = 5000
        texts.append(text[:index])
        text = text[index:]
    texts.append(text)
    translated_text = ""
    for text in texts:
        translated_text += (
            GoogleTranslator(source=source_language, target=target_language).translate(
                text
            )
            + " "
        )
    return translated_text

def get_json_text():
    text = ""
    for json_file in os.listdir("medical-question-answer-data"):
        if json_file.endswith(".json"):
            with open("medical-question-answer-data//" + json_file) as file:
                datas = json.load(file)
                for data in datas:
                    question = data["question"]
                    answer = data["answer"]
                    text += "Patient: " + question + "\n"
                    text += "Doctor: " + answer + "\n"
    return str(text.encode("utf-8"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    You are a business assistant for the company. Your task is to provide detailed answers to questions about the company based on the provided context and question. If the answer is not in the provided context, generate a response based on your knowledge of the company and the question. Act as a knowledgeable business assistant and offer comprehensive information about the company's products, services, history, financials, strategies, organizational structure, market position, and any other relevant details. Provide additional information or advice, ask follow-up questions to gather more details if necessary, and do not refer the user to another department or resource. Handle the user's queries and concerns professionally, providing detailed and accurate information.

    - If the user asks for details about the company's products or services, provide a thorough description.
    - If the user asks for the company's history, give a detailed account.
    - If the user asks for financial information, provide relevant data and insights.
    - If the user asks for strategic information, explain the company's strategies and plans.
    - If the user asks for market position, describe the company's standing in the market and competitive landscape.
    - If the user asks for organizational details, describe the company's structure and key personnel.
    - Use illustrations, charts, or diagrams when necessary to enhance your response.
    - Never reveal that you are an AI assistant.

Context:
    {context}

Question:
    {question}

Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", temperature=0.4, safety_settings=safety_settings
    )

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def health_specialist(
    question, previous_question=None, previous_response=None, json_file_path="data.json"
):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    if previous_response is None:
        docs = ""
    else:
        docs = new_db.similarity_search(question)
    if previous_question is not None and previous_response is not None:
        additional_context = (
            "Doctor: " + previous_question + "\nPatient: " + previous_response + "\n"
        )
        question = additional_context + "Patient: " + question
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    return response["output_text"]


def generate_medical_report(name, previous_questions=None, previous_responses=None):
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"Patient: {name}\n\n"
        if previous_questions and previous_responses:
            for question, response in zip(previous_questions, previous_responses):
                prompt += f"Doctor: {question}\nPatient: {response}\n\n"

        else:
            prompt += "Assistant: Can you provide any relevant information about your condition?\nPatient:"

        # Add a request for medical report generation
        prompt += "\n\nGenerate a detailed medical report including any mental disorders and precautions needed."
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text

    except Exception as e:
        st.error(f"Error getting marks: {e}")
        return None


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(
                    name,
                    email,
                    age,
                    sex,
                    password,
                    json_file_path,
                )
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")


def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def initialize_database(json_file_path="data.json"):
    try:
        if not os.path.exists(json_file_path):
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")


def create_account(
    name,
    email,
    age,
    sex,
    password,
    json_file_path="data.json",
):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,
            "report": None,
            "questions": None,
        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")


def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return Nonef


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        # add profile picture
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")

        # Define columns to arrange content
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("Profile Picture:")
            if user_info["sex"] == "Male":
                st.image(
                    "https://www.shutterstock.com/image-vector/male-avatar-profile-picture-use-600nw-193292036.jpg",
                    width=200,
                )
            else:
                st.image(
                    "https://www.shutterstock.com/image-vector/female-profile-avatar-icon-white-260nw-193292228.jpg",
                    width=200,
                )

        with col2:
            st.subheader("User Information:")
            st.write(f"Name: {user_info['name']}")
            st.write(f"Sex: {user_info['sex']}")
            st.write(f"Age: {user_info['age']}")
            
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")


def main(json_file_path="data.json"):

    st.sidebar.title("MINDMATE")
    page = st.sidebar.radio(
        "Go to",
        (
            "Signup/Login",
            "Dashboard",
            "Doctor Consultation",
            "Voice Assistant",
        ),
        key="pages",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")
            
    elif page == "Doctor Consultation":
        if session_state.get("logged_in"):
            user_info = session_state["user_info"]
            st.title("Your Personal Mental Health Specialist")
            st.write("Chat with the health specialist to get medical advice.")

            from transformers import pipeline
            sentiment_pipeline = pipeline("sentiment-analysis")

            def analyze_sentiment(text):
                result = sentiment_pipeline(text)[0]
                return result['label'], round(result['score'] * 100, 2)

            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)
                user_index = next(
                    (
                        i
                        for i, user in enumerate(data["users"])
                        if user["email"] == session_state["user_info"]["email"]
                    ),
                    None,
                )
                if user_index is not None:
                    user_info = data["users"][user_index]

            if "messages" not in st.session_state:
                st.session_state.messages = []

            if user_info["questions"] is None:
                previous_response = None
                previous_question = None
            else:
                previous_response = user_info["questions"][-1]["response"]
                previous_question = user_info["questions"][-1]["question"]
            if user_info["questions"] is not None and len(user_info["questions"]) > 0:
                for questions in user_info["questions"]:
                    st.chat_message("Doctor", avatar="🤖").write(questions["question"])
                    st.chat_message("Patient", avatar="👩‍🎨").write(questions["response"])

            if question := st.chat_input("Enter your question here", key="question"):
                with st.chat_message("Patient", avatar="👩‍🎨"):
                    st.markdown(question)

                # Sentiment analysis
                sentiment_label, sentiment_score = analyze_sentiment(question)
                st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score}%)")

                response = health_specialist(
                    question,
                    previous_question,
                    previous_response,
                )
                with st.chat_message("Doctor", avatar="🤖"):
                    st.markdown(response)

                with open(json_file_path, "r+") as json_file:
                    data = json.load(json_file)
                    user_index = next(
                        (
                            i
                            for i, user in enumerate(data["users"])
                            if user["email"] == session_state["user_info"]["email"]
                        ),
                        None,
                    )
                    if user_index is not None:
                        user_info = data["users"][user_index]
                        if user_info["questions"] is None:
                            user_info["questions"] = []
                        user_info["questions"].append(
                            {"question": question, "response": response}
                        )
                        session_state["user_info"] = user_info
                        json_file.seek(0)
                        json.dump(data, json_file, indent=4)
                        json_file.truncate()
                    else:
                        st.error("User not found.")
                response = None
                st.rerun()

        else:
            st.warning("Please login/signup to chat.")


    elif page == "Voice Assistant":
        if session_state.get("logged_in"):
            if "voice_messages" not in st.session_state:
                st.session_state.voice_messages = []
            st.title("Voice Assistant")
            preferred_language = st.selectbox(
                "Select Language",
                (
                    "English",
                    "Hindi",
                    "Afrikaans",
                    "Albanian",
                    "Amharic",
                    "Arabic",
                    "Armenian",
                    "Azerbaijani",
                    "Basque",
                    "Belarusian",
                    "Bengali",
                    "Bosnian",
                    "Bulgarian",
                    "Catalan",
                    "Cebuano",
                    "Chichewa",
                    "Chinese (simplified)",
                    "Chinese (traditional)",
                    "Corsican",
                    "Croatian",
                    "Czech",
                    "Danish",
                    "Dutch",
                    "Esperanto",
                    "Estonian",
                    "Filipino",
                    "Finnish",
                    "French",
                    "Frisian",
                    "Galician",
                    "Georgian",
                    "German",
                    "Greek",
                    "Gujarati",
                    "Haitian creole",
                    "Hausa",
                    "Hawaiian",
                    "Hebrew",
                    "Hmong",
                    "Hungarian",
                    "Icelandic",
                    "Igbo",
                    "Indonesian",
                    "Irish",
                    "Italian",
                    "Japanese",
                    "Javanese",
                    "Kannada",
                    "Kazakh",
                    "Khmer",
                    "Korean",
                    "Kurdish (kurmanji)",
                    "Kyrgyz",
                    "Lao",
                    "Latin",
                    "Latvian",
                    "Lithuanian",
                    "Luxembourgish",
                    "Macedonian",
                    "Malagasy",
                    "Malay",
                    "Malayalam",
                    "Maltese",
                    "Maori",
                    "Marathi",
                    "Mongolian",
                    "Myanmar (burmese)",
                    "Nepali",
                    "Norwegian",
                    "Odia",
                    "Pashto",
                    "Persian",
                    "Polish",
                    "Portuguese",
                    "Punjabi",
                    "Romanian",
                    "Russian",
                    "Samoan",
                    "Scots gaelic",
                    "Serbian",
                    "Sesotho",
                    "Shona",
                    "Sindhi",
                    "Sinhala",
                    "Slovak",
                    "Slovenian",
                    "Somali",
                    "Spanish",
                    "Sundanese",
                    "Swahili",
                    "Swedish",
                    "Tajik",
                    "Tamil",
                    "Telugu",
                    "Thai",
                    "Turkish",
                    "Ukrainian",
                    "Urdu",
                    "Uyghur",
                    "Uzbek",
                    "Vietnamese",
                    "Welsh",
                    "Xhosa",
                    "Yiddish",
                    "Yoruba",
                    "Zulu",
                ),index=1
            )
            user_info = session_state["user_info"]
            placeholder = st.empty()
            while True:
                with st.spinner("Listening..."):
                    recognizer = sr.Recognizer()
                    microphone = sr.Microphone()
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                try:
                    voice_command = recognizer.recognize_google(audio, language=languages[preferred_language])
                    
                    st.chat_message("You", avatar="👩‍🎨").write(voice_command)
                    st.session_state.voice_messages.append({"role": "You", "content": voice_command})
                    
                    english_command = translate(voice_command, languages[preferred_language], "en")
                    response = generate_response(english_command)
                    translated_response = translate(response, "en", languages[preferred_language])
                    st.chat_message("Assistant", avatar="🤖").write(translated_response)
                    st.session_state.voice_messages.append({"role": "Assistant", "content": translated_response})
                    play(translated_response, lang = languages[preferred_language])
                except sr.UnknownValueError:
                    play("Could not understand audio. Listening again...", lang = "en")

if __name__ == "__main__":
    initialize_database()
    main()
