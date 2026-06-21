#Starter API Code



from fastapi import FastAPI
import joblib

app = FastAPI()

# Load models
category_model = joblib.load("../models/ticket_category_model.pkl")
priority_model = joblib.load("../models/priority_model.pkl")
resolution_model = joblib.load("../models/resolution_model.pkl")



# -------------------------
# Resolution Generator Function
# PUT HERE
# -------------------------

def generate_resolution(ticket):

    ticket = ticket.lower()

    if "vpn" in ticket:
        return [
            "Check internet connection",
            "Restart VPN client",
            "Verify login credentials",
            "Reconnect to VPN server",
            "Contact network administrator"
        ]
    elif "password" in ticket:
        return [
            "Verify username",
            "Reset password",
            "Check account lock status",
            "Retry login",
            "Contact administrator"
        ]

    elif "email" in ticket:
        return [
            "Check internet connection",
            "Verify mail server",
            "Restart Outlook",
            "Clear cache",
            "Contact IT support"
        ]
    else:
        return [
            "Check system connection",
            "Restart application",
            "Review error logs",
            "Retry operation",
            "Contact support team"
        ]

# -------------------------
# API Routes
# -------------------------



tfidf = joblib.load("../models/tfidf.pkl")

category_encoder = joblib.load("../models/category_encoder.pkl")
priority_encoder = joblib.load("../models/priority_encoder.pkl")


@app.get("/")
def home():

    return {
        "message": "AI ServiceNow Ticket Intelligence API"
    }


@app.post("/predict")
def predict(ticket_description: str):

    vector = tfidf.transform([ticket_description])

    # Category prediction
    category_pred = category_model.predict(vector)

    category = category_encoder.inverse_transform(
        category_pred
    )[0]

    # Priority prediction
    priority_pred = priority_model.predict(vector)

    priority = priority_encoder.inverse_transform(
        priority_pred
    )[0]

    # Resolution time
    resolution_time = resolution_model.predict(
        vector
    )[0]
    # Generate troubleshooting steps
    resolution = generate_resolution(
        ticket_description
    )
    return {

    "ticket": ticket_description,

    "predicted_category": category,

    "predicted_priority": priority,

    "estimated_resolution_time": round(
        resolution_time,
        2
    ),

    "suggested_resolution": resolution
}