# ✅ Install necessary packages
#!pip install --upgrade gradio tensorflow pillow

# ✅ Import Libraries
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image

# ✅ Load Trained Model
model = load_model("Effiicientnetv2b2.keras")
class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# ✅ Image Classification Function
def classify_image(img):
    img = img.resize((260, 260))  # ✅ Change here
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    return f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"


# ✅ Jarvis Chatbot Response Logic
def chatbot_response(question):
    q = (question or "").lower()
    if "hello" in q or "hi" in q:
        return "Hello! I'm Jarvis 😊. How can I assist you?"
    elif "accuracy" in q:
        return "Our garbage classification model achieved a training accuracy of 93.41% and a test accuracy of 93.75% using CNN and EfficientNetv2b2."
    elif "dataset" in q:
        return "We trained on a labeled garbage image dataset with categories like Plastic, Paper, Metal, Glass, and Organic Waste."
    elif "model" in q or "technology" in q:
        return "We used CNN with EfficientNet architecture."
    elif "libraries" in q:
        return "Libraries used: TensorFlow, Keras, NumPy, and Gradio."
    elif "features" in q:
        return "Features: Image upload, webcam input, real-time classification, and an interactive chatbot."
    elif "use case" in q or "application" in q:
        return "Use cases: Smart Bins, City Waste Management, Recycling Plants, Mobile Apps."
    elif "purpose" in q:
        return "The purpose is to automate garbage classification for smart cities and eco-friendly waste sorting."
    elif "requirement" in q:
        return "Requirements: Python, TensorFlow, Gradio, and a labeled image dataset."
    elif "end" in q or "close" in q:
        return "Chat ended. Have a great day! 😊"
    else:
        return "I'm Jarvis 🤖. Ask me about accuracy, dataset, model, features, libraries, or purpose!"

# ✅ Custom CSS for Styling
css_code = """
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f5f5f5;
    color: #333;
}
.navbar, .footer {
    background-color: #4CAF50;
    color: white;
    padding: 15px;
}
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nav-links {
    display: flex;
    gap: 20px;
}
.nav-link {
    cursor: pointer;
    text-decoration: underline;
}
.card, .about-section {
    background-color: #dcedc8;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    color:black;
}
.card:hover, .about-section:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.cards-row {
    display: flex;
    gap:10px;
    overflow-x: auto;
    padding-bottom: 10px;

}
.floating-chat {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #4CAF50;
    color: white;
    border-radius: 50%;
    width: 55px;
    height: 55px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    z-index: 1000;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
#chatbox-container {
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 320px;
    background: white;
    border: 1px solid #ccc;
    border-radius: 10px;
    padding: 15px;
    display: none;
    z-index: 1000;
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.footer-top {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}
.footer-bottom {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 14px;
}
.social-icons{
  display:flex;
}
.social-icons img {
    width: 22px;
    height: 22px;
    margin-left: 10px;
}
"""

# ✅ Gradio UI Blocks
with gr.Blocks(css=css_code) as demo:
    # Navbar
    gr.HTML("""
    <div class='navbar'>
        <div><h2>Garbage Classification AI</h2></div>
        <div class='nav-links'>
            <span class='nav-link' onclick="document.getElementById('garbage').scrollIntoView({behavior:'smooth'});">Garbage Classification</span>
            <span class='nav-link' onclick="document.getElementById('about').scrollIntoView({behavior:'smooth'});">About Us</span>
            <span class='nav-link' onclick="document.getElementById('contact').scrollIntoView({behavior:'smooth'});">Contact</span>
        </div>
    </div>
    """)

    # Garbage Classification Section
    gr.HTML("<div id='garbage'></div>")
    gr.Markdown("## Garbage Classification")
    img_input = gr.Image(type="pil", label="Upload Image or Use Webcam", sources=["upload", "webcam"])
    result_output = gr.Textbox(label="AI Prediction")
    classify_btn = gr.Button("Classify")
    classify_btn.click(fn=classify_image, inputs=img_input, outputs=result_output)

    # About Us Section
    gr.HTML("<div id='about'></div>")
    gr.HTML("""
    <div class='about-section'>
        <h2>About the Project</h2>
        <p>This project focuses on automating garbage classification using Deep Learning techniques.
          A CNN with EfficientNet architecture achieved a training accuracy of 93.41% and a test accuracy of 93.75%.
          Built with Python, TensorFlow, Keras, and NumPy, with a Gradio interface for image upload and real-time classification.
          An AI chatbot named Jarvis is also integrated for user queries.</p>

    </div>
    """)

    # Research Papers Section
    gr.Markdown("## Research Papers")
    gr.HTML("""
    <div class='cards-row '>
        <div class='card'>
            <h4>Multi-Scale CNN Based Garbage Detection</h4>
            <p>Introduces a Multi-Scale CNN (MSCNN) for garbage detection using airborne hyperspectral data.</p>
            <a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782106' target='_blank'>Read More</a>
        </div>
        <div class='card'>
            <h4>Garbage Classification Algorithm</h4>
            <p>Uses an improved MobileNetV3 model to enhance garbage classification accuracy and speed.</p>
            <a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10478519' target='_blank'>Read More</a>
        </div>
        <div class='card'>
            <h4>Automatic Garbage Classification</h4>
            <p>ResNet-34 based deep learning model for automatic garbage classification with high reliability.</p>
            <a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9144549' target='_blank'>Read More</a>
        </div>
    </div>
    """)

    # Contact Section
    gr.HTML("<div id='contact'></div>")
    gr.Markdown("## Contact Us")
    gr.Markdown("For further queries, contact: **miryalavarshitha@gmail.com**")
    # ✅ Jarvis Chatbox (Initially Hidden)
    chatbox_markdown = gr.Markdown("**Jarvis 👋:** Hello! I'm Jarvis. Ask me about the project below:", visible=False)
    jarvis_input = gr.Textbox(placeholder="Type your question...", visible=False)
    jarvis_output = gr.Textbox(label="Jarvis Response", interactive=False, visible=False)
    send_btn = gr.Button("Send", visible=False)
    close_btn = gr.Button("Close", visible=False)
    
    # ✅ Function to open chat (show elements)
    def open_chat():
        return [
            gr.update(visible=True),  # chatbox_markdown
            gr.update(visible=True),  # jarvis_input
            gr.update(visible=True),  # jarvis_output
            gr.update(visible=True),  # send_btn
            gr.update(visible=True),  # close_btn
        ]
    
    # ✅ Function to close chat (hide elements)
    def close_chat():
        return [
            gr.update(visible=False),  # chatbox_markdown
            gr.update(visible=False),  # jarvis_input
            gr.update(visible=False),  # jarvis_output
            gr.update(visible=False),  # send_btn
            gr.update(visible=False),  # close_btn
        ]

    # ✅ Chatbot reply logic
    def send_and_clear(q):
        resp = chatbot_response(q)
        return resp, ""
    
    # ✅ Floating Button (icon style)
    floating_btn = gr.Button("💬", elem_id="floating-chat", visible=True)
    floating_btn.click(fn=open_chat, inputs=None, outputs=[chatbox_markdown, jarvis_input, jarvis_output, send_btn, close_btn])
    
    # ✅ Send button
    send_btn.click(fn=send_and_clear, inputs=jarvis_input, outputs=[jarvis_output, jarvis_input])
    
    # ✅ Close button
    close_btn.click(fn=close_chat, inputs=None, outputs=[chatbox_markdown, jarvis_input, jarvis_output, send_btn, close_btn])

                                                                                                                        

    # Footer
    gr.HTML("""
    <div class='footer'>
        <div class='footer-top'>
            <div>
                <h4>Miryala Varshitha</h4>
                <p>Garbage Classification AI</p>
            </div>
            <div>
                <h4>Social Links</h4>
                <div class='social-icons'>
                    <a href='https://facebook.com' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg'></a>
                    <a href='https://instagram.com/_.miryalavarshitha._' target='_blank'><img src='https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png'></a>
                </div>
            </div>
        </div>
        <div class='footer-bottom'>
            <div>&copy; 2025 Garbage Classification AI. All Rights Reserved.</div>
            <div>Designed by Miryala Varshitha</div>
        </div>
    </div>
    """)

# ✅ Launch Gradio App with Public Link (For Colab)
demo.launch()