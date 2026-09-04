import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faqs = {
    "What is CodeAlpha?": "CodeAlpha is a tech company offering internships and software solutions.",
    "How to apply for internship?": "You can apply through our website www.codealpha.tech",
    "What is the duration of internship?": "Internship duration is 1 month.",
    "Is the internship paid?": "No, it's unpaid but you get certificate and LOR.",
    "What tasks are in AI domain?": "Tasks include FAQ Chatbot, Translation Tool and Object Detection.",
    "How to submit tasks?": "Submit via form shared in WhatsApp group with GitHub and LinkedIn links.",
    "What is AI?": "AI is simulation of human intelligence in machines."
}

questions = list(faqs.keys())
answers = list(faqs.values())
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_answer(user_query):
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    best_match = similarity.argmax()
    if similarity[0][best_match] < 0.3:
        return "Sorry, I didn't understand. Please ask about CodeAlpha internship."
    return answers[best_match]

if __name__ == "__main__":
    print("Chatbot - Type exit to quit")
    while True:
        q = input("You: ")
        if q.lower() == 'exit':
            break
        print("Bot:", get_answer(q))
