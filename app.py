from flask import Flask, Response, request
from flask_cors import CORS
import os
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["OPENAI_API_KEY"]  # Ensure the API key is set in environment variables
)


# Endpoint to handle LLM request
@app.route('/llm/chat', methods=['POST'])
async def get_genes_and_citations():
    data = request.get_json()

    # Extract user input from the POST request
    user_input = data.get('question', '')

    # Generate the prompt
    prompt = (
        f"Answer the following biomedical question in a very specific manner, "
        f"providing only the names of the genes or causes when asked. Do not explain anything extra "
        f"unless specifically asked in the user query. Provide citations to support your answer, "
        f"included with links. Highlight only the main keywords or genes:\n\n{user_input}"
    )

    try:
        # Interact with the LLM model
        completion = client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        # Return the response
        return response, 200


    except Exception as e:
        return str(e), 500

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the LLM API!"

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)