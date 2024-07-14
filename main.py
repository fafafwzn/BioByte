from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama
import time

app = Flask(__name__, template_folder="templates")

def llm_inference(query):
    model_path = "./seallm-7b-chat.q4_k_m.gguf"
    LLM = Llama(model_path=model_path, n_gpu_layers=32, n_threads=6, n_ctx=3584, n_batch=521, verbose=True, temperature=0, cuda_device=0)
    prompt_template = f"""Kamu adalah seorang asisten AI yang akan membantu dokter dalam mengidentifikasi permasalahan pasien. \n
    Kamu akan mendengarkan percakapan antara dokter dengan pasiennya, lalu kamu harus melakukan Name Entity Recognition untuk gejala penyakit pasien. \n
    Contohnya: "Batuk", "Mual", "Bersin", dll. \n
    Setelah melakukan Name Entity Recognition, kamu harus memberikan kemungkinan diagnosa penyakit pasien beserta kemungkinan (dalam persentase) dan alasan akan diagnosa tersebut. \n
    Kamu dapat memberikan lebih dari 1 diagnosa, namun kamu harus menyertakan kemungkinan (dalam persentase) dan alasan untuk masing-masing diagnosa tersebut
    Masukkan kata-kata yang merupakan gejala penyakit tersebut ke dalam format berikut (gantikan huruf xxx): \n\n

    Gejala: xxx \n
    Diagnosa: xxx \n
    Kemungkinan: xxx \n
    Alasan: xxx \n \n

    Sekarang analisalah potongan percakapan berikut! \n

    ```
    {query}
    ```
    """
    a = time.time()
    output = LLM(prompt_template, max_tokens=0)
    b = time.time()
    c = b - a
    return c, output

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    data = request.get_json()
    text = data['text']
    bot_latency, output = llm_inference(text)
    bot_answer = output['choices'][0]['text'].strip()
    response = {
        "answer": bot_answer,
        "latency": bot_latency
    }
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)